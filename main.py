import asyncio
import logging
import re
from typing import Sized
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

import aiohttp
import aiosqlite
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage
from pydantic_settings import BaseSettings

# --------------------- Конфигурация ---------------------
class Settings(BaseSettings):
    API_TOKEN: str
    MISTRAL_API_KEY: str
    HF_API_TOKEN: str | None = None  # теперь Pydantic примет эту переменную

    class Config:
        env_file = ".env"
        model_config = {"extra": "ignore"}


settings = Settings()

# --------------------- Логирование ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------- Глобальные объекты ---------------------
bot = Bot(token=settings.API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

session: aiohttp.ClientSession
db: aiosqlite.Connection
processor: BlipProcessor
model: BlipForConditionalGeneration

# --------------------- Стартовые продукты ---------------------
INITIAL_PRODUCTS = {
    'orange': {'energy': 47, 'proteins': 0.9, 'fat': 0.1, 'carbohydrates': 11.8},
    'apple':  {'energy': 52, 'proteins': 0.3, 'fat': 0.2, 'carbohydrates': 14},
}

# --------------------- Утилиты ---------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

STOP_WORDS = {
    *["a","the","of","with","in","on","and","their","hand","plate","wooden","piece","white","background"],
    *["bread","egg","slice","slices"]
}

def simplify_description(desc: str) -> str:
    words = [w for w in re.findall(r"\w+", desc.lower()) if w not in STOP_WORDS]
    return max(words, key=len) if words else desc.split()[0]

# --------------------- Инициализация ресурсов ---------------------
async def init_resources():
    global session, db, processor, model

    logger.info("Начало инициализации ресурсов")

    # 1) HTTP-сессия
    session = aiohttp.ClientSession()

    # 2) Подключение к БД
    db = await aiosqlite.connect("mealai.db")

    # 3) Загрузка BLIP-модели и процессора в фоновом потоке
    processor_task = asyncio.to_thread(
        BlipProcessor.from_pretrained,
        "Salesforce/blip-image-captioning-base"
    )
    model_task = asyncio.to_thread(
        BlipForConditionalGeneration.from_pretrained,
        "Salesforce/blip-image-captioning-base"
    )
    processor, model = await asyncio.gather(processor_task, model_task)
    model.to("cpu")

    # 4) Создание таблиц
    await db.execute("""
        CREATE TABLE IF NOT EXISTS mapping (
            blip_desc TEXT PRIMARY KEY,
            dish_name TEXT
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS local_nutrition (
            name TEXT PRIMARY KEY,
            energy REAL, proteins REAL, fat REAL, carbohydrates REAL
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS mistral_cache (
            prompt TEXT PRIMARY KEY,
            response TEXT
        )
    """)
    # Сидинг стартовых продуктов
    for name, vals in INITIAL_PRODUCTS.items():
        await db.execute(
            "REPLACE INTO local_nutrition VALUES (?,?,?,?,?)",
            (name, vals['energy'], vals['proteins'], vals['fat'], vals['carbohydrates'])
        )
    await db.commit()

    logger.info("Ресурсы инициализированы")

# --------------------- FSM-состояния ---------------------
class FeedbackState(StatesGroup):
    waiting_confirmation = State()

# --------------------- Сервисы ---------------------
async def generate_caption(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

async def call_mistral(prompt: str) -> str:
    # 1) Сначала пытаемся взять ответ из кеша
    async with db.execute("SELECT response FROM mistral_cache WHERE prompt=?", (prompt,)) as cur:
        row = await cur.fetchone()
        if row:
            return row[0]

    # 2) Если нет в кеше – делаем запрос
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "open-mistral-7b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300
    }
    for attempt in range(2):
        try:
            async with session.post(url, headers=headers, json=body, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                # Сохраняем в кеш
                await db.execute(
                    "REPLACE INTO mistral_cache VALUES (?,?)",
                    (prompt, text)
                )
                await db.commit()
                return text
        except Exception as e:
            logger.warning("Mistral попытка %d не удалась: %s", attempt+1, e)
            await asyncio.sleep(1)
    return ""

async def get_dish_name(desc: str) -> str:
    prompt = (
        f"На русском, коротко и по делу: по описанию изображения «{desc}» "
        "скажи название блюда или продукта одним-двумя словами."
    )
    name = await call_mistral(prompt)
    if not name or len(name.split()) > 3 or re.search(r"[A-Za-z]", name):
        return simplify_description(desc)
    return name

async def get_nutrition(name: str) -> str:
    prompt = (
        f"На русском, строго в формате «Калории: X ккал; Белки: Y г; Жиры: Z г; Углеводы: W г»: "
        f"дай пищевую ценность на 100 г блюда «{name}»."
    )
    return await call_mistral(prompt)

async def get_recommendation(nutr: str) -> str:
    prompt = (
        f"На основе данных «{nutr}», напиши краткую практичную рекомендацию "
        "по употреблению этого продукта, не более 3 предложений. "
        "Фокусируйся на пользе и умеренности."
    )
    return await call_mistral(prompt)

# --------------------- Работа с БД ---------------------
async def fetch_mapping(desc: str) -> str | None:
    norm = normalize(desc)
    async with db.execute("SELECT dish_name FROM mapping WHERE blip_desc=?", (norm,)) as cur:
        row = await cur.fetchone()
        return row[0] if row else None

async def save_mapping(desc: str, dish: str):
    norm = normalize(desc)
    await db.execute(
        "REPLACE INTO mapping (blip_desc, dish_name) VALUES (?,?)",
        (norm, dish)
    )
    await db.commit()

async def fetch_local_nutrition(name: str) -> dict | None:
    async with db.execute(
        "SELECT energy, proteins, fat, carbohydrates FROM local_nutrition WHERE name=?", (name.lower(),)
    ) as cur:
        row = await cur.fetchone()
        if not row:
            return None
        return {
            "energy": row[0],
            "proteins": row[1],
            "fat": row[2],
            "carbohydrates": row[3]
        }

# --------------------- Хендлеры ---------------------
@dp.message(CommandStart())
async def cmd_start(msg: Message):
    await msg.answer("Привет! Пришли фото блюда или продукта — я определю его, скажу ккал и дам рекомендацию.")

@dp.message(F.photo)
async def handle_photo(msg: Message, state: FSMContext):
    # 1) Скачать и прочитать байты
    try:
        f = await bot.get_file(msg.photo[-1].file_id)
        buf = await bot.download_file(f.file_path)
        img_bytes = buf.read()
    except Exception as e:
        logger.error("Ошибка загрузки фото: %s", e)
        return await msg.reply("Ошибка загрузки фото, попробуйте ещё раз.")

    # 2) Сгенерировать подпись
    desc = await generate_caption(img_bytes)
    if not desc:
        return await msg.reply("Не удалось описать изображение.")
    await msg.reply(f"Описание: {desc}")

    # 3) Определить название
    dish = await fetch_mapping(desc)
    is_new = False
    if not dish:
        dish = await get_dish_name(desc)
        await save_mapping(desc, dish)
        is_new = True
    await msg.reply(f"Определено как: {dish}")

    # 4) Пищевая ценность
    local = await fetch_local_nutrition(dish)
    if local:
        nutr_str = (
            f"Калории: {local['energy']} ккал; Белки: {local['proteins']} г; "
            f"Жиры: {local['fat']} г; Углеводы: {local['carbohydrates']} г"
        )
    else:
        nutr_str = await get_nutrition(dish)
    await msg.reply(f"Пищевая ценность на 100 г:\n{nutr_str}")

    # 5) Рекомендация
    rec = await get_recommendation(nutr_str)
    await msg.reply(f"Рекомендация: {rec}")

    # 6) Обратная связь по названию
    if is_new:
        await msg.reply("Это правильное название? Ответьте «Да» или введите корректное.")
        await state.update_data(blip_desc=desc)
        await state.set_state(FeedbackState.waiting_confirmation)

@dp.message(FeedbackState.waiting_confirmation)
async def feedback_handler(msg: Message, state: FSMContext):
    data = await state.get_data()
    desc = data.get("blip_desc")
    ans = msg.text.strip()
    if ans.lower() == "да":
        await msg.reply("Спасибо, оставлю как есть.")
    else:
        await save_mapping(desc, ans)
        await msg.reply(f"Запомнил: «{desc}» — это «{ans}».")
    await state.clear()

# --------------------- Запуск ---------------------
async def main():
    await init_resources()
    await bot.delete_webhook(drop_pending_updates=True)
    try:
        await dp.start_polling(bot)
    finally:
        await session.close()
        await db.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")

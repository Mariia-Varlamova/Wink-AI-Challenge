from langchain_community.llms import YandexGPT
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
import json

# -------------------------------
# Настройки
# -------------------------------
MAX_SCENE_WORDS = 300
CHROMA_PERSIST_DIRECTORY = "chroma_db"  # Папка для хранения базы данных Chroma

# Загрузка переменных из .env файла
load_dotenv()

# -------------------------------
# Подключение к YandexGPT
# -------------------------------
llm = YandexGPT(
    api_key=os.getenv('YANDEX_API_KEY'),
    folder_id=os.getenv('YANDEX_FOLDER_ID')
)

# Проверка загрузки ключей
if not os.getenv('YANDEX_API_KEY'):
    raise ValueError("YANDEX_API_KEY не найден в переменных окружения")

# -------------------------------
# Функции
# -------------------------------
def clean_text(text: str) -> str:
    """Удаляет лишние пробелы и переносы строк"""
    return re.sub(r'\s+', ' ', text.strip())

def split_text_for_embeddings(text: str):
    """Разбиваем текст на куски для эмбеддингов"""
    sentences = re.split(r'(?<=[.!?])\s+(?=[А-ЯA-Z])', clean_text(text))
    chunks, current = [], []
    for s in sentences:
        current.append(s)
        if len(" ".join(current).split()) >= MAX_SCENE_WORDS:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def ask_yandex_gpt(prompt: str) -> str:
    """Отправка промпта в YandexGPT"""
    # Передаём список строк
    result = llm.generate([prompt])
    # Берём текст первой генерации
    return result.generations[0][0].text

def run_rag_pipeline(text: str, question: str, persist_directory: str = CHROMA_PERSIST_DIRECTORY):
    """RAG-подход: поиск релевантного контекста и генерация ответа с использованием ChromaDB"""
    chunks = split_text_for_embeddings(text)
    
    # Используем выбранные эмбеддинги
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Создаем Chroma векторное хранилище
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    # Сохраняем базу данных
    vectorstore.persist()

    # Находим 3 самых релевантных фрагмента
    relevant_chunks = vectorstore.similarity_search(question, k=3)
    context = "\n".join([c.page_content for c in relevant_chunks])

    prompt = (
        "Используй только этот контекст для ответа. "
        "Раздели текст на сцены по локациям и основным событиям. "
        "Собери действия и диалоги по персонажам. "
        "Не добавляй ничего от себя. "
        "Верни результат строго в виде JSON-списка, каждая сцена - это текст самой сцены. "
        "Каждый объект должен иметь ключ 'text_scene' с полным текстом сцены.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос:\n{question}\nОтвет:"
    )

    answer = ask_yandex_gpt(prompt)

    # Попробуем извлечь JSON
    try:
        json_part = re.search(r'\[.*\]', answer, re.DOTALL).group(0)
        scenes = json.loads(json_part)
    except Exception as e:
        print(f"⚠️ Не удалось получить JSON — fallback: делим по длине и без персонажей. Ошибка: {e}")
        scenes = [{"text_scene": chunk} for chunk in chunks]

    return scenes

def load_existing_chroma_db(persist_directory: str = CHROMA_PERSIST_DIRECTORY):
    """Загружает существующую Chroma базу данных"""
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

def add_detailed_scene_numbers(scenes):
    """Добавляет нумерацию и расширенную структуру"""
    result = {}
    for i, scene in enumerate(scenes, 1):
        scene_key = f"scene{i}"
        
        # Если сцена уже имеет структуру, сохраняем её
        if isinstance(scene, dict):
            enhanced_scene = scene.copy()
        else:
            # Если сцена - просто текст, создаём структуру
            enhanced_scene = {"text_scene": scene}
        
        # Добавляем мета-информацию
        enhanced_scene["scene_number"] = i
        enhanced_scene["scene_id"] = scene_key
        enhanced_scene["word_count"] = len(enhanced_scene.get("text_scene", "").split())
        
        result[scene_key] = enhanced_scene
    
    return result

def save_detailed_scenes(scenes, filename="scenes_detailed.json"):
    """Сохраняет сцены с расширенной структурой"""
    detailed_scenes = add_detailed_scene_numbers(scenes)
    
    # Добавляем общую статистику
    output = {
        "metadata": {
            "total_scenes": len(detailed_scenes),
            "total_words": sum(scene.get("word_count", 0) for scene in detailed_scenes.values()),
            "chroma_db_directory": CHROMA_PERSIST_DIRECTORY
        },
        "scenes": detailed_scenes
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    return output

def clear_chroma_db(persist_directory: str = CHROMA_PERSIST_DIRECTORY):
    """Очищает Chroma базу данных"""
    import shutil
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"✅ Chroma база данных в '{persist_directory}' очищена")

# -------------------------------
# Пример использования
# -------------------------------
if __name__ == "__main__":
    text = """
    Ночной город был почти пуст. На улице шёл дождь, и свет фонарей отражался в мокром асфальте. Анна стояла у входа в старое здание и ждала. Её зонт сломался, капли стекали по волосам, но она не двигалась, погружённая в свои мысли. Вдали послышался шум мотора — подъехала чёрная машина. Из неё вышел мужчина в длинном пальто. Он посмотрел на Анну и молча подошёл.

Внутри здания пахло пылью и старой бумагой. Лестница скрипела под ногами, и каждый звук эхом разносился по пустым коридорам. Мужчина и Анна поднялись на второй этаж, где за закрытой дверью доносились тихие голоса. Они прислушались, но услышали только шелест бумаги и дождь за окнами.

В соседней комнате лампы мигнули, и тень прошла по стене. Анна заметила на столе странный конверт. Она медленно подошла к нему и открыла — внутри был старый ключ и записка: «Только для тебя». Мужчина сделал знак, что нужно идти дальше.

Они вышли на крышу здания. Ветер срывал дождевые капли с крыш и кружил их в воздухе. Город лежал под ними, как гигантская мокрая карта с огоньками фонарей. Анна посмотрела вниз и ощутила странное чувство свободы, смешанное с тревогой.

Вдруг вдалеке раздался громкий звук: мотор мотоцикла, быстро приближающегося к зданию. Мужчина схватил Анну за руку и потянул к люку выхода на пожарную лестницу. Они спрыгнули вниз, чувствуя, как холодный дождь бьёт по лицам, и растворились в темноте улиц, словно растворились в самом городе.
    """

    question = "Раздели текст на сцены, собери действия и диалоги по персонажам. Не придумывай ничего."
    
    # Очищаем предыдущую базу (опционально)
    # clear_chroma_db()
    
    scenes = run_rag_pipeline(text, question)

    print("\nРазделённые сцены:\n")
    for i, scene in enumerate(scenes, 1):
        print(f"Сцена {i}: {scene}\n")

    # Сохраняем с расширенной структурой
    detailed_output = save_detailed_scenes(scenes, "scenes_detailed.json")
    print("\nРасширенная структура с метаданными:")
    print(json.dumps(detailed_output, ensure_ascii=False, indent=2))
    
    print(f"\n✅ Chroma база данных сохранена в папке: {CHROMA_PERSIST_DIRECTORY}")
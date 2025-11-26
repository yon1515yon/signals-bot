import os
import time
from newsapi import NewsApiClient
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import date, timedelta
import logging
from app.config import settings  

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)



sentiment_pipeline = None

MODEL_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

LOCK_FILE_PATH = os.path.join(MODEL_CACHE_DIR, '.finbert.lock')


def get_sentiment_pipeline():
    """
    Инициализирует модель только один раз, используя файловый замок
    для предотвращения гонки состояний в многопроцессорной среде.
    """
    global sentiment_pipeline
    if sentiment_pipeline is not None:
        return sentiment_pipeline

    try:
        with open(LOCK_FILE_PATH, "x") as lock_file:
            print(">>> Создан файловый замок. Начинаю загрузку модели FinBERT... Это может занять несколько минут.")

            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

            print(">>> Модель FinBERT успешно загружена.")

    except FileExistsError:
        print(">>> Другой процесс уже загружает модель. Ожидаю завершения...")
        while os.path.exists(LOCK_FILE_PATH):
            time.sleep(5) 
        print(">>> Замок снят. Загружаю модель с диска.")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    except Exception as e:
        print(f"!!! Не удалось загрузить модель sentiment-analysis: {e}.")
        if os.path.exists(LOCK_FILE_PATH):
            os.remove(LOCK_FILE_PATH) 
        return None
    finally:
        if os.path.exists(LOCK_FILE_PATH):
            os.remove(LOCK_FILE_PATH)

    return sentiment_pipeline


def get_sentiment_score(ticker: str, company_name: str) -> tuple[float | None, int]:
    """
    Получает новости и вычисляет средний балл сентимента.
    Возвращает кортеж: (средний балл, количество новостей).
    Средний балл: от -1 (очень негативно) до 1 (очень позитивно).
    """

    pipeline_instance = get_sentiment_pipeline()

    if not settings.NEWS_API_KEY or not sentiment_pipeline:
        return None, 0

    newsapi = NewsApiClient(api_key=settings.NEWS_API_KEY)
    from_date = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    query = f'"{company_name}" OR {ticker}'

    try:
        all_articles = newsapi.get_everything(
            q=query, language="ru,en", sort_by="relevancy", from_param=from_date, page_size=20
        )
    except Exception as e:
        print(f"Ошибка при получении новостей для {ticker}: {e}")
        return None, 0

    articles = all_articles.get("articles", [])
    if not articles:
        return 0.0, 0

    scores = []
    for article in articles:
        title = article.get("title")
        if not title or "[Removed]" in title:
            continue

        try:
            result = pipeline_instance(title)[0]
            if result["label"] == "positive":
                scores.append(result["score"])
            elif result["label"] == "negative":
                scores.append(-result["score"])
        except Exception:
            continue

    if not scores:
        return 0.0, len(articles)

    avg_score = sum(scores) / len(scores)
    return avg_score, len(articles)

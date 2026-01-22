import structlog
from sqlalchemy import text
from app.core.celery_app import celery_app
from app.db.session import session_scope
from app.services.sentiment import get_sentiment_score

logger = structlog.get_logger()

@celery_app.task(
    name="app.worker.tasks.nlp.analyze_sentiment", 
    queue="nlp_queue",  
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3},
    rate_limit="10/m"  
)
def analyze_sentiment_task(ticker: str, company_name: str):
    """
    Задача для анализа новостного фона.
    Загружает тяжелую модель FinBERT (только в этом воркере).
    """
    log = logger.bind(task="analyze_sentiment", ticker=ticker)
    log.info("Starting sentiment analysis")

    try:
        score, news_count = get_sentiment_score(ticker, company_name)

        if news_count == 0:
            log.info("No news found")
            return

        # Обновляем БД
        with session_scope() as db:
            stmt = text("""
                UPDATE tracked_tickers 
                SET sentiment_score = :score, last_updated_at = NOW() 
                WHERE ticker = :ticker
            """)
            db.execute(stmt, {"score": score, "ticker": ticker})
        
        log.info("Sentiment updated", score=score, count=news_count)

    except Exception as e:
        log.error("Sentiment analysis failed", error=str(e))
        raise e
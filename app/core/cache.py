import redis
from app.config import settings

redis_client = redis.Redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5
)

def get_redis():
    """Возвращает экземпляр клиента Redis."""
    return redis_client
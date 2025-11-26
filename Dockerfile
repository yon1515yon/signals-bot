FROM python:3.12-slim

# Устанавливаем системные зависимости
# Добавил gcc и build-essential, так как для некоторых ML библиотек (numpy, pandas, talib) они нужны при сборке
RUN apt-get update && \
    apt-get install -y postgresql-client libgdbm-compat-dev git gcc build-essential libpq-dev --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Переменные окружения для ML библиотек (чтобы не жрали все ядра)
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    # ВАЖНО: Добавляем текущую директорию в путь Python, чтобы он видел пакет app
    PYTHONPATH=/app

# Сначала копируем requirements для кэширования слоя pip install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ВЕСЬ код. 
# Так как структура теперь /app/app/main.py..., просто COPY . . сработает корректно,
# если Dockerfile лежит в корне проекта (рядом с папкой app).
COPY . .

# Создаем папки, если их нет (хотя docker-compose volume их перекроет, это good practice)
RUN mkdir -p /app/models_storage /app/hf_cache

# Если у тебя есть wait-for-it.sh, убедись, что он тоже скопирован
# RUN chmod +x /app/wait-for-it.sh 

# Команда по умолчанию (можно переопределить в docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
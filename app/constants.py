# --- Тикеры ---
LIQUID_TICKERS = [
    "SBER",
    "GAZP",
    "LKOH",
    "GMKN",
    "NVTK",
    "YNDX",
    "ROSN",
    "PLZL",
    "MGNT",
    "TATN",
    "SNGS",
    "MTSS",
    "VTBR",
    "ALRS",
    "CHMF",
    "NLMK",
]

# --- Внешние URL ---
CBR_KEY_RATE_URL = "http://www.cbr.ru/scripts/KeyRate_xml.asp"
ROSSTAT_INFLATION_URL = "https://rosstat.gov.ru/storage/mediabank/ipc_mes_gg_28-05-2024.csv"

# --- Параметры ML (из model.py) ---
LSTM_PARAMS = {"hidden_size": 128, "num_layers": 2, "dropout": 0.3, "n_head": 4}

# --- Настройки Сканера ---
MIN_LIQUIDITY_RUB = 1_000_000
MAX_SPREAD_PCT = 1.5
WALL_VOLUME_RATIO = 0.02  # 2%

# --- Настройки Ghost ---
GHOST_WINDOW = 6
GHOST_VOL_RATIO = 3.0
GHOST_PRICE_CHANGE_LIMIT = 0.4

# --- Настройки Darkpool ---
DARKPOOL_VOL_SPIKE = 3.5
DARKPOOL_VOLATILITY_QUANTILE = 0.20

# --- ML Hyperparameters ---
ML_CONFIG = {
    # Архитектура модели
    "HIDDEN_SIZE": 128,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.3,
    "N_HEAD": 4,
    "OUTPUT_SIZE": 4,  # close, volume, rsi, log_return
    # Параметры обучения (Global Model)
    "BATCH_SIZE_GLOBAL": 128,
    "EPOCHS_GLOBAL": 20,
    "LEARNING_RATE_GLOBAL": 0.001,
    # Параметры дообучения (Fine-Tuning)
    "BATCH_SIZE_FINE_TUNE": 64,
    "EPOCHS_FINE_TUNE": 30,
    "EPOCHS_TRANSFER": 5,  # Если веса уже есть
    "LEARNING_RATE_FINE_TUNE": 0.001,
    "LEARNING_RATE_TRANSFER": 0.0001,  # Меньше, чтобы не сломать веса
    # Подготовка данных
    "TRAIN_WINDOW": 60,  # Длина последовательности (дней)
    # Функция потерь
    "FOCAL_LOSS_GAMMA": 1.0,
    "FOCAL_LOSS_WEIGHTS": [2.0, 1.0, 0.5, 1.5],  # Веса для выходов
}

ROBUST_FACTOR = 0.55
BACKTEST_Z_THRESHOLD = 1.0 
RISK_PER_TRADE = 0.95   

AI_SL_SAFETY_MARGIN = 1.1  
ATR_SL_MULTIPLIER = 2.0    
LEVEL_SL_MARGIN = 0.995    
FIXED_SL_PERCENT = 0.95    
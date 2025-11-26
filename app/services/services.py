from datetime import datetime, timedelta
from io import StringIO
from xml.parsers.expat import ExpatError

import pandas as pd
import requests
import xmltodict
from celery.utils.log import get_task_logger

from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now

from app.config import settings
from app.constants import CBR_KEY_RATE_URL, ROSSTAT_INFLATION_URL

logger = get_task_logger(__name__)


def get_all_russian_stocks():
    """Получает список всех российских акций, торгуемых на MOEX (СИНХРОННАЯ ВЕРСИЯ)."""
    stocks_info = []
    with Client(settings.TINKOFF_API_TOKEN) as client:
        instruments = client.instruments.shares()
        for instrument in instruments.instruments:
            if instrument.class_code == "TQBR" and instrument.currency == "rub":
                stocks_info.append({"name": instrument.name, "ticker": instrument.ticker, "figi": instrument.figi})
    return stocks_info


def get_historical_data(
    figi: str, days: int, interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_DAY
) -> pd.DataFrame:
    """Загружает исторические данные (свечи) для указанного FIGI (СИНХРОННАЯ ВЕРСИЯ)."""
    candles_data = []
    with Client(settings.TINKOFF_API_TOKEN) as client:
        for candle in client.get_all_candles(
            figi=figi,
            from_=now() - timedelta(days=days),
            interval=interval,
        ):
            candles_data.append(
                {
                    "time": candle.time,
                    "open": float(f"{candle.open.units}.{candle.open.nano}"),
                    "close": float(f"{candle.close.units}.{candle.close.nano}"),
                    "high": float(f"{candle.high.units}.{candle.high.nano}"),
                    "low": float(f"{candle.low.units}.{candle.low.nano}"),
                    "volume": candle.volume,
                }
            )
    if not candles_data:
        return pd.DataFrame()
    return pd.DataFrame(candles_data).sort_values(by="time")


def get_order_book(figi: str) -> dict:
    """Получает стакан ордеров для указанного FIGI."""
    try:
        with Client(settings.TINKOFF_API_TOKEN) as client:
            order_book = client.market_data.get_order_book(figi=figi, depth=20)
            return {
                "bids": [
                    {"price": q.price.units + q.price.nano / 1e9, "quantity": q.quantity} for q in order_book.bids
                ],
                "asks": [
                    {"price": q.price.units + q.price.nano / 1e9, "quantity": q.quantity} for q in order_book.asks
                ],
            }
    except Exception as e:
        print(f"Не удалось получить стакан для FIGI {figi}: {e}")
        return {"bids": [], "asks": []}


def find_imoex_future_figi() -> str | None:
    """
    Находит FIGI самого ликвидного (ближайшего) фьючерса на Индекс МосБиржи (IMOEX).
    """

    with Client(settings.TINKOFF_API_TOKEN) as client:
        futures = client.instruments.futures()
        imoex_futures = []

        target_basic_asset = "IMOEX"

        for f in futures.instruments:
            if f.basic_asset == target_basic_asset and f.expiration_date > now():
                imoex_futures.append(f)

        if not imoex_futures:
            print(f"Не найдено ни одного активного фьючерса с базовым активом '{target_basic_asset}'.")
            return None

        closest_future = sorted(imoex_futures, key=lambda f: f.expiration_date)[0]

        print(f"Найден актуальный фьючерс на IMOEX: {closest_future.ticker} (FIGI: {closest_future.figi})")
        return closest_future.figi


def get_cbr_key_rate() -> pd.DataFrame:
    """
    Парсит историю ключевой ставки с сайта ЦБ РФ через их XML API.
    """
    logger.info("Загрузка истории ключевой ставки с сайта ЦБ РФ...")
    url = CBR_KEY_RATE_URL
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        try:
            data = xmltodict.parse(response.content)
            records = data["KeyRate"]["Record"]
        except (ExpatError, KeyError) as e:
            logger.error(f"Ошибка парсинга XML от ЦБ: {e}. Возможно, ответ поврежден.")
            return pd.DataFrame()

        rates = []
        for record in records:
            rates.append({"date": datetime.strptime(record["@Date"], "%d.%m.%Y"), "key_rate": float(record["Rate"])})

        df = pd.DataFrame(rates).sort_values("date")
        df.rename(columns={"date": "time"}, inplace=True)
        logger.info(f"Успешно загружено {len(df)} записей о ключевой ставке.")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Сетевая ошибка при загрузке ключевой ставки ЦБ: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Не удалось загрузить ключевую ставку ЦБ: {e}")
        return pd.DataFrame()


def get_rosstat_inflation() -> pd.DataFrame:
    """
    Парсит данные по инфляции (ИПЦ, год к году) с сайта Росстата.
    """
    logger.info("Загрузка данных по инфляции (ИПЦ) с сайта Росстата...")
    url = ROSSTAT_INFLATION_URL
    try:
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

        response = requests.get(url, timeout=15, verify=False)
        response.raise_for_status()
        # --------------------------------------------------------------------------

        csv_data = response.content.decode("cp1251")
        df = pd.read_csv(StringIO(csv_data), sep=";", skiprows=2)

        df_melted = df.melt(id_vars=[df.columns[0]], var_name="year", value_name="cpi")
        df_melted.rename(columns={df.columns[0]: "month_name"}, inplace=True)
        df_melted.dropna(subset=["cpi"], inplace=True)
        df_melted["year"] = pd.to_numeric(df_melted["year"], errors="coerce")
        df_melted.dropna(subset=["year"], inplace=True)
        df_melted["year"] = df_melted["year"].astype(int)
        month_map = {
            "январь": 1,
            "февраль": 2,
            "март": 3,
            "апрель": 4,
            "май": 5,
            "июнь": 6,
            "июль": 7,
            "август": 8,
            "сентябрь": 9,
            "октябрь": 10,
            "ноябрь": 11,
            "декабрь": 12,
        }
        df_melted["month"] = df_melted["month_name"].str.lower().map(month_map)
        df_melted.dropna(subset=["month"], inplace=True)
        df_melted["month"] = df_melted["month"].astype(int)
        df_melted["time"] = pd.to_datetime(df_melted[["year", "month"]].assign(day=1))
        df_final = df_melted[["time", "cpi"]].copy()
        df_final["cpi"] = df_final["cpi"].astype(str).str.replace(",", ".").astype(float) - 100
        logger.info(f"Успешно загружено {len(df_final)} записей об инфляции.")
        return df_final.sort_values("time")

    except requests.exceptions.RequestException as e:
        logger.error(f"Сетевая ошибка при загрузке данных по инфляции: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Не удалось обработать данные по инфляции: {e}")
        return pd.DataFrame()


def get_combined_macro_data() -> pd.DataFrame:
    """
    Объединяет данные по ключевой ставке и инфляции в один DataFrame.
    """
    logger.info("Объединение макроэкономических данных...")
    key_rate_df = get_cbr_key_rate()
    inflation_df = get_rosstat_inflation()

    if key_rate_df.empty or inflation_df.empty:
        logger.warning("Не удалось загрузить один или несколько источников макро-данных.")
        return pd.DataFrame()

    combined = pd.merge_asof(
        inflation_df.sort_values("time"), key_rate_df.sort_values("time"), on="time", direction="backward"
    )
    return combined


def get_cross_asset_data(days: int) -> pd.DataFrame:
    """
    Загружает данные по коррелирующим активам: фьючерсам на USD/RUB и Золото.
    Динамически находит актуальные FIGI.
    """
    logger.info("Загрузка данных по коррелирующим активам (фьючерсы Si, Gold)...")

    usd_data = pd.DataFrame()
    gold_data = pd.DataFrame()

    with Client(settings.TINKOFF_API_TOKEN) as client:
        usd_future_figi = _find_future_figi_by_ticker(client, "Si")
        gold_future_figi = _find_future_figi_by_ticker(client, "GD")
        # ------------------------------------------------

    if usd_future_figi:
        temp_usd = get_historical_data(figi=usd_future_figi, days=days)
        if not temp_usd.empty:
            usd_data = temp_usd[["time", "close"]].rename(columns={"close": "usd_rub_close"})
        else:
            logger.warning("Не удалось загрузить данные по фьючерсу USD/RUB (Si).")

    if gold_future_figi:
        temp_gold = get_historical_data(figi=gold_future_figi, days=days)
        if not temp_gold.empty:
            gold_data = temp_gold[["time", "close"]].rename(columns={"close": "gold_close"})
        else:
            logger.warning("Не удалось загрузить данные по фьючерсу на Золото (GD).")

    if usd_data.empty and gold_data.empty:
        logger.error("Не удалось загрузить данные ни по одному из кросс-активов.")
        return pd.DataFrame()

    if not usd_data.empty and not gold_data.empty:
        merged_df = pd.merge(usd_data, gold_data, on="time", how="outer")
    elif not usd_data.empty:
        merged_df = usd_data
    else:
        merged_df = gold_data

    merged_df.sort_values("time", inplace=True)
    merged_df.ffill(inplace=True)

    return merged_df


def _find_future_figi_by_ticker(client, ticker_base: str) -> str | None:
    """Находит FIGI для ближайшего активного фьючерса по базе тикера (e.g., 'Si', 'GD')."""
    try:
        futures = client.instruments.futures().instruments
        active_futures = [f for f in futures if f.ticker.startswith(ticker_base) and f.expiration_date > now()]
        if not active_futures:
            logger.warning(f"Не найдено активных фьючерсов для базы '{ticker_base}'.")
            return None

        closest_future = sorted(active_futures, key=lambda f: f.expiration_date)[0]
        logger.info(
            f"Найден актуальный фьючерс для '{ticker_base}': {closest_future.ticker} (FIGI: {closest_future.figi})"
        )
        return closest_future.figi
    except Exception as e:
        logger.error(f"Ошибка при поиске фьючерса для '{ticker_base}': {e}")
        return None

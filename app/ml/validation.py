import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

class StockDataSchema(pa.SchemaModel):
    """
    Схема валидации исторических данных перед обучением.
    Гарантирует, что данные чистые и типизированные.
    """
    time: Series[pd.DatetimeTZDtype] = pa.Field(coerce=True)
    open: Series[float] = pa.Field(ge=0.0)
    high: Series[float] = pa.Field(ge=0.0)
    low: Series[float] = pa.Field(ge=0.0)
    close: Series[float] = pa.Field(ge=0.0)
    volume: Series[float] = pa.Field(ge=0.0)

    # Кастомные проверки
    @pa.check("high")
    def high_gt_low(cls, series: Series[float], df: pd.DataFrame) -> Series[bool]:
        """High должен быть >= Low"""
        return series >= df["low"]

    class Config:
        coerce = True  
        strict = False 


def validate_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Валидирует и очищает данные.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    df = df.dropna()

    if len(df) < 100:
        raise ValueError(f"Not enough data points: {len(df)}")

    try:
        validated_df = StockDataSchema.validate(df)
        return validated_df
    except pa.errors.SchemaError as e:
        raise ValueError(f"Data validation failed: {e}")
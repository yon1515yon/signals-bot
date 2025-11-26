import pytest
from models import TrackedTicker
from sqlalchemy.exc import IntegrityError


def test_create_ticker(db_session):
    # Создаем
    ticker = TrackedTicker(ticker="TEST", figi="BBG000", name="Test Stock")
    db_session.add(ticker)
    db_session.commit()

    # Читаем
    read_ticker = db_session.query(TrackedTicker).filter_by(ticker="TEST").first()
    assert read_ticker.name == "Test Stock"

    # Проверка уникальности
    ticker2 = TrackedTicker(ticker="TEST", figi="BBG111")  # Дубликат
    db_session.add(ticker2)
    with pytest.raises(IntegrityError):
        db_session.commit()

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to NeuroVision API!", "documentation": "/docs"}


def test_get_signals_empty(client, db_session):
    # Пока база пустая, должен вернуть пустой список
    response = client.get("/signals")
    assert response.status_code == 200
    assert response.json() == []

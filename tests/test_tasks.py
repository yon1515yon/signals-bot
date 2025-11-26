from app.worker.tasks import celery_app

def test_celery_config():
    # Проверяем, что роутинг настроен
    routes = celery_app.conf.task_routes
    assert 'tasks.run_global_model_training' in routes
    assert routes['tasks.run_global_model_training']['queue'] == 'ml_training'
    
    # Проверяем расписание
    schedule = celery_app.conf.beat_schedule
    assert 'scan-order-book-breakouts' in schedule
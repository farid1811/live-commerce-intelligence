"""Analytics Controller"""
from flask import Blueprint, render_template, current_app
from src.repositories.model_repository import ModelRepository
from src.services.prediction_service import PredictionService

analytics_bp = Blueprint('analytics', __name__, url_prefix='/analytics')


@analytics_bp.route('/')
def index():
    df = current_app.config.get('DATASET')
    model_info = {'name': 'No model trained', 'type': 'N/A', 'metrics': {}}
    dataset_shape = (0, 0)

    if df is not None:
        dataset_shape = df.shape

    try:
        repo = ModelRepository(registry_dir=current_app.config['MODEL_DIR'])
        active = repo.get_active_model()
        if active:
            model_info = {
                'name': active.get('name', 'N/A'),
                'type': active.get('model_type', 'N/A'),
                'data_type': active.get('data_type', 'N/A'),
                'metrics': active.get('metrics', {}),
                'filename': active.get('filename', ''),
                'timestamp': active.get('timestamp', 'N/A')
            }
    except Exception:
        pass

    return render_template(
        'analytics.html',
        page_title='Analytics',
        model_info=model_info,
        dataset_rows=dataset_shape[0],
        dataset_cols=dataset_shape[1]
    )

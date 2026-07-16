"""Reports Controller"""
from flask import Blueprint, render_template, current_app
from src.repositories.model_repository import ModelRepository

reports_bp = Blueprint('reports', __name__, url_prefix='/reports')


@reports_bp.route('/')
def index():
    active_model = {'name': 'No model trained', 'model_type': 'N/A', 'metrics': {}}

    try:
        repo = ModelRepository(registry_dir=current_app.config['MODEL_DIR'])
        m = repo.get_active_model()
        if m:
            active_model = {
                'name': m.get('name', 'N/A'),
                'model_type': m.get('model_type', 'N/A'),
                'data_type': m.get('data_type', 'N/A'),
                'metrics': m.get('metrics', {}),
                'filename': m.get('filename', ''),
                'timestamp': m.get('timestamp', 'N/A')
            }
    except Exception:
        pass

    df = current_app.config.get('DATASET')
    dataset_info = {'rows': 0, 'has_data': False}
    if df is not None:
        dataset_info = {'rows': len(df), 'has_data': True}

    return render_template(
        'reports.html',
        page_title='Reports',
        active_model=active_model,
        dataset_info=dataset_info
    )

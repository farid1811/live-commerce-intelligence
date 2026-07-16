"""Settings Controller"""
import os
from flask import Blueprint, render_template, current_app
from src.repositories.model_repository import ModelRepository

settings_bp = Blueprint('settings', __name__, url_prefix='/settings')


@settings_bp.route('/')
def index():
    df = current_app.config.get('DATASET')
    dataset_rows = len(df) if df is not None else 0

    try:
        repo = ModelRepository(registry_dir=current_app.config['MODEL_DIR'])
        models_list = repo.list_models()
        model_count = len(models_list)
    except Exception:
        models_list = []
        model_count = 0

    system_info = {
        'model_count': model_count,
        'dataset_rows': dataset_rows,
        'model_dir': current_app.config.get('MODEL_DIR', 'N/A'),
        'upload_dir': current_app.config.get('UPLOAD_FOLDER', 'N/A'),
        'default_dataset': current_app.config.get('DATASET_PATH', 'N/A'),
        'product_version': '1.0.0',
        'flask_env': 'Development',
        'sgd_constraints': 'theta ≥ 0, bias ≥ 0'
    }

    return render_template(
        'settings.html',
        page_title='Settings',
        system_info=system_info,
        models_list=models_list
    )

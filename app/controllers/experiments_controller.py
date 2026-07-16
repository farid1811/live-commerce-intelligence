"""Experiments Controller"""
import json
from flask import Blueprint, render_template, current_app
from src.repositories.model_repository import ModelRepository

experiments_bp = Blueprint('experiments', __name__, url_prefix='/experiments')


@experiments_bp.route('/')
def index():
    try:
        repo = ModelRepository(registry_dir=current_app.config['MODEL_DIR'])
        models_list = repo.list_models()
    except Exception:
        models_list = []

    models_json = json.dumps(models_list)
    return render_template(
        'experiments.html',
        page_title='Model Experiments',
        models_list=models_list,
        models_json=models_json
    )

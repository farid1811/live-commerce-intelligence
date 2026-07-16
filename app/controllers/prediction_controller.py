"""Prediction Controller"""
from flask import Blueprint, render_template, current_app
from src.repositories.model_repository import ModelRepository

prediction_bp = Blueprint('prediction', __name__, url_prefix='/prediction')


@prediction_bp.route('/')
def index():
    try:
        repo = ModelRepository(registry_dir=current_app.config['MODEL_DIR'])
        models_list = repo.list_models()
    except Exception:
        models_list = []

    return render_template(
        'prediction.html',
        page_title='Prediction Center',
        models_list=models_list
    )

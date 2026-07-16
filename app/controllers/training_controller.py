"""Training Controller"""
from flask import Blueprint, render_template, current_app
from src.repositories.model_repository import ModelRepository

training_bp = Blueprint('training', __name__, url_prefix='/training')


@training_bp.route('/')
def index():
    try:
        repo = ModelRepository(registry_dir=current_app.config['MODEL_DIR'])
        models_history = repo.list_models()
    except Exception:
        models_history = []

    return render_template(
        'training.html',
        page_title='Training Center',
        models_history=models_history
    )

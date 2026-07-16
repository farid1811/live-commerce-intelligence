"""StreamAnalytica Flask Application Factory"""
import os
import sys

# Ensure project root is in path for src/ imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from flask import Flask
from app.config import Config


def create_app(config_class=Config):
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')
    app.config.from_object(config_class)

    # Ensure storage directories exist
    os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Load default dataset at startup
    _load_default_dataset(app)

    # Register page controllers (Blueprints)
    from app.controllers.dashboard_controller import dashboard_bp
    from app.controllers.prediction_controller import prediction_bp
    from app.controllers.analytics_controller import analytics_bp
    from app.controllers.intelligence_controller import intelligence_bp
    from app.controllers.training_controller import training_bp
    from app.controllers.dataset_controller import dataset_bp
    from app.controllers.experiments_controller import experiments_bp
    from app.controllers.reports_controller import reports_bp
    from app.controllers.settings_controller import settings_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(intelligence_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(experiments_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(settings_bp)

    # Register API blueprint
    from app.api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api/v1')

    return app


def _load_default_dataset(app):
    """Load the default dataset into app config at startup."""
    try:
        from src.repositories.dataset_repository import DatasetRepository
        repo = DatasetRepository(app.config['DATASET_PATH'])
        df = repo.load_data()
        app.config['DATASET'] = df
        app.config['DATASET_LOADED'] = True
        print(f"[StreamAnalytica] Dataset loaded: {len(df)} sessions")
    except Exception as e:
        app.config['DATASET'] = None
        app.config['DATASET_LOADED'] = False
        print(f"[StreamAnalytica] Warning: Could not load default dataset: {e}")

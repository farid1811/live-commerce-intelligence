"""StreamAnalytica REST API — /api/v1"""
from flask import Blueprint

api_bp = Blueprint('api', __name__)

# Import and register sub-blueprints
from app.api.dashboard_api import dashboard_api_bp
from app.api.prediction_api import prediction_api_bp
from app.api.training_api import training_api_bp
from app.api.dataset_api import dataset_api_bp
from app.api.models_api import models_api_bp
from app.api.reports_api import reports_api_bp
from app.api.analytics_api import analytics_api_bp

api_bp.register_blueprint(dashboard_api_bp)
api_bp.register_blueprint(prediction_api_bp)
api_bp.register_blueprint(training_api_bp)
api_bp.register_blueprint(dataset_api_bp)
api_bp.register_blueprint(models_api_bp)
api_bp.register_blueprint(reports_api_bp)
api_bp.register_blueprint(analytics_api_bp)

"""Prediction API"""
import numpy as np
from flask import Blueprint, jsonify, request, current_app
from src.services.prediction_service import PredictionService
from src.repositories.model_repository import ModelRepository

prediction_api_bp = Blueprint('prediction_api', __name__, url_prefix='/predict')


@prediction_api_bp.route('', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        durasi = float(data.get('durasi', 8.0))
        penonton = int(data.get('penonton', 100))
        model_filename = data.get('model_filename')

        repo = ModelRepository(registry_dir=current_app.config['MODEL_DIR'])
        pred_service = PredictionService(repo)

        model_data = None
        if model_filename:
            try:
                model_data = repo.load_model(model_filename)
            except Exception:
                model_data = None

        result = pred_service.predict(durasi, penonton, model_data)

        # Convert numpy types to Python native for JSON serialization
        def to_native(obj):
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        return jsonify(to_native(result))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

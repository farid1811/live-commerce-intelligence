"""Dashboard API"""
import numpy as np
from flask import Blueprint, jsonify, current_app
from src.services.prediction_service import PredictionService
from src.services.analytics_service import AnalyticsService
from src.repositories.model_repository import ModelRepository

dashboard_api_bp = Blueprint('dashboard_api', __name__, url_prefix='/dashboard')


@dashboard_api_bp.route('/summary', methods=['GET'])
def summary():
    try:
        df = current_app.config.get('DATASET')
        if df is None:
            return jsonify({'error': 'No dataset loaded'}), 400

        avg_duration = float(df['Durasi_Jam'].mean())
        avg_viewers = int(round(df['Penonton Aktif'].mean()))
        avg_sales = float(round(df['Produk Terjual'].mean(), 2))
        total_sessions = len(df)

        repo = ModelRepository(registry_dir=current_app.config['MODEL_DIR'])
        pred_service = PredictionService(repo)
        active_model = repo.get_active_model()
        pred_result = pred_service.predict(avg_duration, avg_viewers, active_model)

        growth_info = AnalyticsService.calculate_growth_rates(df)

        return jsonify({
            'avg_duration': round(avg_duration, 2),
            'avg_viewers': avg_viewers,
            'avg_sales': avg_sales,
            'total_sessions': total_sessions,
            'prediction': {
                'value': pred_result['prediction_correct'],
                'lower': pred_result['lower_bound'],
                'upper': pred_result['upper_bound'],
                'confidence': pred_result['confidence'],
                'risk_level': pred_result['risk_level'],
                'recommendation': pred_result['recommendation']
            },
            'growth': {
                'sales_growth': round(float(growth_info['latest_sales_growth']), 2)
                    if growth_info['latest_sales_growth'] == growth_info['latest_sales_growth'] else 0,
                'viewers_growth': round(float(growth_info['latest_viewers_growth']), 2)
                    if growth_info['latest_viewers_growth'] == growth_info['latest_viewers_growth'] else 0,
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

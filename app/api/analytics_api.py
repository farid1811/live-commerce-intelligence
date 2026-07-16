"""Analytics API — chart data computation endpoints"""
import numpy as np
from flask import Blueprint, jsonify, current_app
from src.services.prediction_service import PredictionService
from src.services.data_service import DataService
from src.services.analytics_service import AnalyticsService
from src.repositories.model_repository import ModelRepository

analytics_api_bp = Blueprint('analytics_api', __name__, url_prefix='/analytics')


def _get_repo():
    return ModelRepository(registry_dir=current_app.config['MODEL_DIR'])


@analytics_api_bp.route('/surface', methods=['GET'])
def surface_data():
    """Compute 3D regression surface data points."""
    try:
        df = current_app.config.get('DATASET')
        if df is None:
            return jsonify({'error': 'No dataset loaded'}), 400

        repo = _get_repo()
        pred_service = PredictionService(repo)
        active_model = repo.get_active_model()
        if active_model is None:
            active_model = pred_service.get_fallback_model()

        data_type = active_model.get('data_type', 'All Data')
        df_proc = DataService.get_preprocessed_dataset(df, data_type)
        X = df_proc[['Durasi_Jam', 'Penonton Aktif']].values
        y = df_proc['Produk Terjual'].values

        # Build predicted values for scatter
        y_preds = []
        for i in range(len(X)):
            res = pred_service.predict(float(X[i, 0]), int(X[i, 1]), active_model)
            y_preds.append(res['raw_correct'])
        y_preds = np.array(y_preds)
        residuals = (y - y_preds).tolist()

        # Build surface grid (20x20)
        dur_grid = np.linspace(float(X[:, 0].min()), float(X[:, 0].max()), 20).tolist()
        view_grid = np.linspace(float(X[:, 1].min()), float(X[:, 1].max()), 20).tolist()
        sales_mesh = []
        for v in view_grid:
            row = []
            for d in dur_grid:
                res = pred_service.predict(d, int(v), active_model)
                row.append(round(res['raw_correct'], 2))
            sales_mesh.append(row)

        return jsonify({
            'actual': {'x': X[:, 0].tolist(), 'y': X[:, 1].tolist(), 'z': y.tolist()},
            'predicted': {'x': X[:, 0].tolist(), 'y': X[:, 1].tolist(), 'z': y_preds.tolist()},
            'residuals': residuals,
            'surface': {'dur_grid': dur_grid, 'view_grid': view_grid, 'sales_mesh': sales_mesh},
            'model_name': active_model.get('name', 'Default Model'),
            'metrics': active_model.get('metrics', {})
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analytics_api_bp.route('/moving-average', methods=['GET'])
def moving_average():
    """Moving average chart data."""
    try:
        df = current_app.config.get('DATASET')
        if df is None:
            return jsonify({'error': 'No dataset loaded'}), 400

        df_ma = AnalyticsService.calculate_moving_averages(df, window=7)
        return jsonify({
            'sessions': list(range(1, len(df_ma) + 1)),
            'actual_sales': df_ma['Produk Terjual'].tolist(),
            'ma_sales': df_ma['Moving_Avg_Sales'].round(2).tolist(),
            'actual_viewers': df_ma['Penonton Aktif'].tolist(),
            'ma_viewers': df_ma['Moving_Avg_Viewers'].round(2).tolist(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analytics_api_bp.route('/outliers', methods=['GET'])
def outliers():
    """Outlier session detection."""
    try:
        df = current_app.config.get('DATASET')
        if df is None:
            return jsonify({'error': 'No dataset loaded'}), 400

        df_anom = AnalyticsService.detect_outliers_regression_residuals(df)
        outliers_df = df_anom[df_anom['Is_Outlier_Session'] == True].copy()
        
        result = []
        for _, row in outliers_df.head(10).iterrows():
            result.append({
                'bulan': str(row.get('Bulan', 'N/A')),
                'durasi': round(float(row['Durasi_Jam']), 2),
                'viewers': int(row['Penonton Aktif']),
                'actual': int(row['Produk Terjual']),
                'expected': round(float(row['Expected_Sales']), 1),
                'residual': round(float(row['Residual']), 2),
                'type': str(row['Outlier_Type'])
            })
        return jsonify({'outliers': result, 'total_flagged': len(outliers_df)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

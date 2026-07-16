"""Dashboard Controller"""
from flask import Blueprint, render_template, current_app
from src.services.analytics_service import AnalyticsService
from src.services.prediction_service import PredictionService
from src.repositories.model_repository import ModelRepository

dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/')


@dashboard_bp.route('/')
def index():
    df = current_app.config.get('DATASET')
    summary = {}
    growth_info = {}
    prediction_result = {}
    active_model_name = 'No Model Trained'

    if df is not None:
        summary = {
            'total_sessions': len(df),
            'avg_duration': round(float(df['Durasi_Jam'].mean()), 2),
            'avg_viewers': int(round(df['Penonton Aktif'].mean())),
            'avg_sales': round(float(df['Produk Terjual'].mean()), 2),
            'max_sales': int(df['Produk Terjual'].max()),
            'min_sales': int(df['Produk Terjual'].min()),
        }
        try:
            growth_data = AnalyticsService.calculate_growth_rates(df)
            sg = growth_data.get('latest_sales_growth', 0)
            vg = growth_data.get('latest_viewers_growth', 0)
            growth_info = {
                'sales_growth': round(float(sg), 2) if sg == sg else 0.0,
                'viewers_growth': round(float(vg), 2) if vg == vg else 0.0,
                'latest_month': growth_data.get('latest_month', 'N/A')
            }
        except Exception:
            growth_info = {'sales_growth': 0.0, 'viewers_growth': 0.0, 'latest_month': 'N/A'}

        try:
            repo = ModelRepository(registry_dir=current_app.config['MODEL_DIR'])
            pred_service = PredictionService(repo)
            active_model = repo.get_active_model()
            if active_model:
                active_model_name = active_model.get('name', 'Default Model')
            result = pred_service.predict(summary['avg_duration'], summary['avg_viewers'], active_model)
            prediction_result = {
                'prediction': result.get('prediction_correct', 0),
                'lower_bound': result.get('lower_bound', 0),
                'upper_bound': result.get('upper_bound', 0),
                'confidence': result.get('confidence', 50),
                'risk_level': result.get('risk_level', 'Medium'),
                'recommendation': result.get('recommendation', ''),
                'interpretation': result.get('interpretation', ''),
                'model_name': result.get('model_name', 'Default')
            }
        except Exception as e:
            prediction_result = {
                'prediction': 0, 'lower_bound': 0, 'upper_bound': 0,
                'confidence': 50, 'risk_level': 'Medium',
                'recommendation': '', 'interpretation': str(e), 'model_name': 'N/A'
            }

    return render_template(
        'dashboard.html',
        page_title='Executive Dashboard',
        summary=summary,
        growth_info=growth_info,
        prediction_result=prediction_result,
        active_model_name=active_model_name
    )

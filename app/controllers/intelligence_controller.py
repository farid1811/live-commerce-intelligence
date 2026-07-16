"""Intelligence Controller"""
import json
from flask import Blueprint, render_template, current_app
from src.services.analytics_service import AnalyticsService

intelligence_bp = Blueprint('intelligence', __name__, url_prefix='/intelligence')


@intelligence_bp.route('/')
def index():
    df = current_app.config.get('DATASET')
    growth_info = {'sales_growth': 0.0, 'viewers_growth': 0.0, 'latest_month': 'N/A'}
    seasonality_json = '[]'

    if df is not None:
        try:
            g = AnalyticsService.calculate_growth_rates(df)
            sg = g.get('latest_sales_growth', 0)
            vg = g.get('latest_viewers_growth', 0)
            growth_info = {
                'sales_growth': round(float(sg), 2) if sg == sg else 0.0,
                'viewers_growth': round(float(vg), 2) if vg == vg else 0.0,
                'latest_month': g.get('latest_month', 'N/A')
            }
        except Exception:
            pass

        try:
            seas_df = AnalyticsService.analyze_seasonality(df)
            seas_df = seas_df.fillna(0)
            seas_df['Bulan'] = seas_df['Bulan'].astype(str)
            seasonality_json = json.dumps(seas_df.round(2).to_dict(orient='records'))
        except Exception:
            seasonality_json = '[]'

    return render_template(
        'intelligence.html',
        page_title='Business Intelligence',
        growth_info=growth_info,
        seasonality_json=seasonality_json
    )

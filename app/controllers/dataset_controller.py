"""Dataset Controller"""
from flask import Blueprint, render_template, current_app
from src.services.data_service import DataService

dataset_bp = Blueprint('dataset', __name__, url_prefix='/dataset')


@dataset_bp.route('/')
def index():
    df = current_app.config.get('DATASET')
    dataset_info = {'rows': 0, 'columns': [], 'has_data': False}
    stats = []

    if df is not None:
        dataset_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'has_data': True
        }
        try:
            stats_df = DataService.get_summary_statistics(df)
            stats = stats_df.round(4).to_dict(orient='records')
        except Exception:
            stats = []

    return render_template(
        'dataset.html',
        page_title='Dataset Manager',
        dataset_info=dataset_info,
        stats=stats
    )

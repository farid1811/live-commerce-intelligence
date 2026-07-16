"""Dataset API — upload, stats, correlation"""
import os
import pandas as pd
from flask import Blueprint, jsonify, request, current_app
from src.services.data_service import DataService
from src.repositories.dataset_repository import DatasetRepository

dataset_api_bp = Blueprint('dataset_api', __name__, url_prefix='/dataset')


@dataset_api_bp.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Empty filename'}), 400

        # Save to uploads directory
        upload_dir = current_app.config['UPLOAD_FOLDER']
        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)

        # Load and validate
        repo = DatasetRepository(filepath)
        df = repo.load_data()
        is_valid, msg = repo.validate_schema(df)

        if not is_valid:
            os.remove(filepath)
            return jsonify({'success': False, 'message': msg}), 400

        # Update app dataset
        current_app.config['DATASET'] = df
        current_app.config['DATASET_LOADED'] = True

        return jsonify({
            'success': True,
            'message': f'Dataset loaded successfully: {len(df)} sessions',
            'rows': len(df),
            'columns': list(df.columns)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@dataset_api_bp.route('/stats', methods=['GET'])
def stats():
    try:
        df = current_app.config.get('DATASET')
        if df is None:
            return jsonify({'error': 'No dataset loaded'}), 400

        stats_df = DataService.get_summary_statistics(df)
        preview_records = df.head(10).fillna('').to_dict(orient='records')

        return jsonify({
            'stats': stats_df.round(4).to_dict(orient='records'),
            'preview': preview_records,
            'rows': len(df),
            'columns': list(df.columns)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@dataset_api_bp.route('/correlation', methods=['GET'])
def correlation():
    try:
        df = current_app.config.get('DATASET')
        if df is None:
            return jsonify({'error': 'No dataset loaded'}), 400

        corr = DataService.get_correlation_matrix(df)
        return jsonify({
            'labels': list(corr.columns),
            'matrix': corr.round(4).values.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

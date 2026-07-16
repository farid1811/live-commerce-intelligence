"""Models Registry API"""
import os
from flask import Blueprint, jsonify, request, current_app
from src.repositories.model_repository import ModelRepository

models_api_bp = Blueprint('models_api', __name__, url_prefix='/models')


def _get_repo():
    return ModelRepository(registry_dir=current_app.config['MODEL_DIR'])


@models_api_bp.route('/list', methods=['GET'])
def list_models():
    try:
        repo = _get_repo()
        models = repo.list_models()
        return jsonify({'models': models, 'count': len(models)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@models_api_bp.route('/activate', methods=['POST'])
def activate_model():
    try:
        data = request.get_json(force=True)
        filename = data.get('filename')
        if not filename:
            return jsonify({'success': False, 'message': 'No filename provided'}), 400
        repo = _get_repo()
        repo.set_active_model(filename)
        return jsonify({'success': True, 'message': f'Model {filename} is now active'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@models_api_bp.route('/delete/<filename>', methods=['DELETE'])
def delete_model(filename):
    try:
        repo = _get_repo()
        filepath = os.path.join(repo.registry_dir, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'Model not found'}), 404
        os.remove(filepath)
        return jsonify({'success': True, 'message': f'Model {filename} deleted'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@models_api_bp.route('/purge', methods=['DELETE'])
def purge_all():
    try:
        repo = _get_repo()
        count = 0
        for f in os.listdir(repo.registry_dir):
            if f.endswith('.pkl'):
                os.remove(os.path.join(repo.registry_dir, f))
                count += 1
        return jsonify({'success': True, 'message': f'Purged {count} models'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

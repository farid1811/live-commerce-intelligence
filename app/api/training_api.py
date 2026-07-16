"""Training API — SSE real-time training endpoint"""
import uuid
import queue
import threading
import json
import time
from flask import Blueprint, jsonify, request, Response, stream_with_context, current_app
from src.services.training_service import TrainingService
from src.services.data_service import DataService
from src.repositories.model_repository import ModelRepository

training_api_bp = Blueprint('training_api', __name__, url_prefix='/training')

# Global session store: {session_id: queue.Queue()}
_training_sessions: dict = {}
_session_lock = threading.Lock()


def _get_repo(app):
    return ModelRepository(registry_dir=app.config['MODEL_DIR'])


@training_api_bp.route('/start', methods=['POST'])
def start_training():
    """Start a training run and return a session_id for SSE streaming."""
    try:
        data = request.get_json(force=True)
        model_type = data.get('model_type', 'Linear')
        data_type = data.get('data_type', 'All Data')
        learning_rate = float(data.get('learning_rate', 0.01))
        max_epochs = int(data.get('max_epochs', 1000))
        tolerance = float(data.get('tolerance', 0.001))

        df = current_app.config.get('DATASET')
        if df is None:
            return jsonify({'error': 'No dataset loaded. Please upload a dataset first.'}), 400

        # Create a session
        session_id = str(uuid.uuid4())
        event_queue = queue.Queue()
        with _session_lock:
            _training_sessions[session_id] = event_queue

        # Capture app context for thread
        app = current_app._get_current_object()

        # Run training in a background thread
        def run_training():
            try:
                repo = _get_repo(app)
                training_service = TrainingService(repo)
                
                start_time = time.time()
                epoch_count = [0]

                def epoch_callback(epoch, theta, bias, loss, rmse, delta):
                    epoch_count[0] = epoch
                    elapsed = time.time() - start_time
                    avg_per_epoch = elapsed / epoch if epoch > 0 else 0.01
                    eta = avg_per_epoch * (max_epochs - epoch)
                    progress_pct = min(99, int((epoch / max_epochs) * 100))

                    event = {
                        'type': 'progress',
                        'epoch': epoch,
                        'loss': round(float(loss), 8),
                        'rmse': round(float(rmse), 8),
                        'delta': round(float(delta), 8),
                        'progress_pct': progress_pct,
                        'eta_seconds': round(eta, 1),
                        'elapsed': round(elapsed, 1)
                    }
                    event_queue.put(event)

                model_data = training_service.train_model(
                    df, model_type, data_type,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    tolerance=tolerance,
                    callback=epoch_callback
                )

                # Final event
                event_queue.put({
                    'type': 'complete',
                    'model_name': model_data.get('name'),
                    'model_type': model_data.get('model_type'),
                    'data_type': model_data.get('data_type'),
                    'metrics': model_data.get('metrics', {}),
                    'total_epochs': epoch_count[0],
                    'elapsed': round(time.time() - start_time, 2)
                })

            except Exception as e:
                event_queue.put({'type': 'error', 'message': str(e)})
            finally:
                # Signal end of stream
                event_queue.put(None)

        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()

        return jsonify({'status': 'started', 'session_id': session_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_api_bp.route('/stream/<session_id>', methods=['GET'])
def stream_training(session_id):
    """SSE endpoint — streams training progress events to browser."""
    with _session_lock:
        event_queue = _training_sessions.get(session_id)

    if event_queue is None:
        return jsonify({'error': 'Session not found'}), 404

    def generate():
        try:
            while True:
                try:
                    event = event_queue.get(timeout=60)  # 60s timeout
                    if event is None:
                        yield f"data: {json.dumps({'type': 'end'})}\n\n"
                        break
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get('type') in ('complete', 'error'):
                        yield f"data: {json.dumps({'type': 'end'})}\n\n"
                        break
                except queue.Empty:
                    # Send keepalive ping
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        finally:
            # Cleanup session
            with _session_lock:
                _training_sessions.pop(session_id, None)

    response = Response(
        stream_with_context(generate()),
        mimetype='text/event-stream'
    )
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection'] = 'keep-alive'
    return response


@training_api_bp.route('/history', methods=['GET'])
def training_history():
    """Return all registered models."""
    try:
        repo = _get_repo(current_app._get_current_object())
        models = repo.list_models()
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

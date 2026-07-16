/* ====================================================
   StreamAnalytica — training.js
   SSE Training Manager — Real-time SGD Progress
   ==================================================== */

'use strict';

class TrainingManager {
  constructor() {
    this.eventSource = null;
    this.epochData   = [];
    this.lossData    = [];
    this.startTime   = null;
    this.isTraining  = false;
    this.sessionId   = null;
  }

  async startTraining(config) {
    if (this.isTraining) {
      showToast('A training session is already in progress.', 'warning');
      return;
    }

    // Reset state
    this.epochData  = [];
    this.lossData   = [];
    this.startTime  = Date.now();
    this.isTraining = true;

    // Show progress panel
    const panel = document.getElementById('trainingProgressPanel');
    if (panel) panel.style.display = 'block';

    // Reset UI elements
    this._resetUI();
    this._addLog('INFO', 'Initiating training session...');
    this._addLog('INFO', `Model: ${config.model_type} | Data: ${config.data_type} | LR: ${config.learning_rate} | Epochs: ${config.max_epochs}`);

    try {
      // POST to start training → get session_id
      const res = await postJSON('/api/v1/training/start', config);
      this.sessionId = res.session_id;
      this._addLog('INFO', `Session ID: ${this.sessionId.substring(0, 8)}... — Stream connected.`);

      // Connect SSE stream
      this._connectStream(this.sessionId);
    } catch (err) {
      this.isTraining = false;
      showToast(`Training failed to start: ${err.message}`, 'error');
      this._addLog('ERROR', err.message);
    }
  }

  _connectStream(sessionId) {
    this.eventSource = new EventSource(`/api/v1/training/stream/${sessionId}`);

    this.eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this._handleEvent(data);
      } catch (e) {
        console.error('SSE parse error:', e);
      }
    };

    this.eventSource.onerror = () => {
      this.eventSource.close();
      if (this.isTraining) {
        this.isTraining = false;
        showToast('Training stream disconnected unexpectedly.', 'error');
        this._addLog('ERROR', 'Connection to training stream lost.');
      }
    };
  }

  _handleEvent(data) {
    switch (data.type) {
      case 'progress':
        this._onProgress(data);
        break;
      case 'complete':
        this._onComplete(data);
        break;
      case 'error':
        this._onError(data);
        break;
      case 'end':
        if (this.eventSource) this.eventSource.close();
        break;
      case 'ping':
        // keepalive, ignore
        break;
    }
  }

  _onProgress(data) {
    const { epoch, loss, rmse, delta, progress_pct, eta_seconds, elapsed } = data;

    // Update progress bar
    const bar = document.getElementById('trainingProgressBar');
    if (bar) {
      bar.style.width = `${progress_pct}%`;
      bar.textContent = `${progress_pct}%`;
    }

    // Update metrics
    this._setText('currentLoss',  loss.toExponential(4));
    this._setText('currentRmse',  rmse.toFixed(6));
    this._setText('currentDelta', delta.toExponential(4));
    this._setText('currentEpoch', epoch.toLocaleString());
    this._setText('etaDisplay',   this._formatEta(eta_seconds));
    this._setText('elapsedDisplay', `${elapsed.toFixed(1)}s`);

    // Update loss chart (every 5 epochs to avoid too frequent redraws)
    this.epochData.push(epoch);
    this.lossData.push(loss);

    if (epoch % 5 === 0 || epoch === 1) {
      this._updateLossCurve();
    }

    // Add log line (every 25 epochs)
    if (epoch % 25 === 0 || epoch === 1) {
      this._addLog('EPOCH', `[${String(epoch).padStart(4)}] Loss: ${loss.toExponential(4)} | RMSE: ${rmse.toFixed(6)} | Δ: ${delta.toExponential(3)}`);
    }
  }

  _onComplete(data) {
    this.isTraining = false;
    if (this.eventSource) this.eventSource.close();

    const { model_name, metrics, total_epochs, elapsed } = data;

    // Fill progress to 100%
    const bar = document.getElementById('trainingProgressBar');
    if (bar) { bar.style.width = '100%'; bar.textContent = '100%'; }

    // Show final metrics
    this._addLog('DONE', `Training complete after ${total_epochs} epochs in ${elapsed.toFixed(2)}s`);
    this._addLog('DONE', `Model: ${model_name}`);
    if (metrics) {
      this._addLog('DONE', `R²: ${metrics.r2?.toFixed(4)} | MAE: ${metrics.mae?.toFixed(4)} | RMSE: ${metrics.rmse?.toFixed(4)} | MAPE: ${metrics.mape?.toFixed(2)}%`);
    }

    showToast(`✓ Model trained: ${model_name}`, 'success', 6000);

    // Show final metrics section
    const metricsPanel = document.getElementById('finalMetricsPanel');
    if (metricsPanel && metrics) {
      metricsPanel.style.display = 'block';
      this._setText('finalR2',   metrics.r2?.toFixed(4));
      this._setText('finalMae',  metrics.mae?.toFixed(4));
      this._setText('finalRmse', metrics.rmse?.toFixed(4));
      this._setText('finalMape', metrics.mape?.toFixed(2) + '%');
    }

    // Reload model registry table
    setTimeout(() => this._reloadModelRegistry(), 1000);
  }

  _onError(data) {
    this.isTraining = false;
    if (this.eventSource) this.eventSource.close();
    showToast(`Training error: ${data.message}`, 'error');
    this._addLog('ERROR', data.message);
  }

  _updateLossCurve() {
    const chartEl = document.getElementById('lossCurveChart');
    if (chartEl && typeof renderLossCurve === 'function') {
      renderLossCurve('lossCurveChart', this.epochData, this.lossData);
    }
  }

  async _reloadModelRegistry() {
    try {
      const data = await getJSON('/api/v1/models/list');
      const tbody = document.getElementById('modelRegistryBody');
      if (!tbody || !data.models) return;

      if (data.models.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted py-4">No models trained yet.</td></tr>';
        return;
      }

      tbody.innerHTML = data.models.map(m => {
        const metrics = m.metrics || {};
        const isActive = m.is_active;
        return `<tr>
          <td>${isActive ? '<span class="badge badge-active"><i class="bi bi-check2-circle me-1"></i>Active</span>' : '<span class="badge bg-light text-secondary">Idle</span>'}</td>
          <td><strong>${m.name || m.filename}</strong></td>
          <td><span class="badge badge-${(m.model_type||'').toLowerCase()}">${m.model_type || 'N/A'}</span></td>
          <td>${m.data_type || 'N/A'}</td>
          <td>${metrics.r2 != null ? metrics.r2.toFixed(4) : '—'}</td>
          <td>${metrics.mae != null ? metrics.mae.toFixed(4) : '—'}</td>
          <td>${metrics.rmse != null ? metrics.rmse.toFixed(4) : '—'}</td>
          <td>${m.timestamp || 'N/A'}</td>
          <td>
            <button class="btn btn-sm btn-outline-primary" onclick="activateModel('${m.filename}')">Activate</button>
          </td>
        </tr>`;
      }).join('');
    } catch (e) {
      console.error('Failed to reload registry:', e);
    }
  }

  _addLog(level, message) {
    const log = document.getElementById('trainingLog');
    if (!log) return;
    const now = new Date().toLocaleTimeString('en-GB', { hour12: false });
    const colorClass = level === 'ERROR' ? 'text-danger' :
                       level === 'DONE'  ? 'log-loss' :
                       level === 'EPOCH' ? 'log-epoch' : '';
    const line = document.createElement('div');
    line.innerHTML = `<span style="color:#475569">[${now}]</span> <span class="${colorClass}">[${level}]</span> ${message}`;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  }

  _resetUI() {
    const bar = document.getElementById('trainingProgressBar');
    if (bar) { bar.style.width = '0%'; bar.textContent = '0%'; }
    ['currentLoss','currentRmse','currentDelta','currentEpoch','etaDisplay','elapsedDisplay']
      .forEach(id => this._setText(id, '—'));
    const log = document.getElementById('trainingLog');
    if (log) log.innerHTML = '';
    const mp = document.getElementById('finalMetricsPanel');
    if (mp) mp.style.display = 'none';
  }

  _setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  _formatEta(seconds) {
    if (seconds < 5) return '< 5s';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${s}s`;
  }
}

// ── Global instance ────────────────────────────────
window.trainingManager = new TrainingManager();

// ── Helper: activate model ──────────────────────────
async function activateModel(filename) {
  try {
    const res = await postJSON('/api/v1/models/activate', { filename });
    if (res.success) {
      showToast(`Model activated: ${filename}`, 'success');
      window.trainingManager._reloadModelRegistry();
    } else {
      showToast(res.message || 'Failed to activate', 'error');
    }
  } catch (e) {
    showToast(e.message, 'error');
  }
}

window.activateModel = activateModel;

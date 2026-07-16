/* ====================================================
   StreamAnalytica — main.js
   Global JS utilities, toast, skeleton, HTTP helpers
   ==================================================== */

'use strict';

// ── Toast Notifications ──────────────────────────────
const TOAST_ICONS = {
  success: '<i class="bi bi-check-circle-fill" style="color:var(--success)"></i>',
  error:   '<i class="bi bi-x-circle-fill" style="color:var(--danger)"></i>',
  warning: '<i class="bi bi-exclamation-triangle-fill" style="color:var(--warning)"></i>',
  info:    '<i class="bi bi-info-circle-fill" style="color:var(--primary)"></i>',
};

function showToast(message, type = 'success', duration = 4000) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast-item ${type}`;
  toast.innerHTML = `
    ${TOAST_ICONS[type] || TOAST_ICONS.info}
    <span style="flex:1">${message}</span>
    <button onclick="this.parentElement.remove()" 
      style="background:none;border:none;cursor:pointer;color:var(--text-muted);font-size:14px;padding:0;margin-left:6px;">
      <i class="bi bi-x"></i>
    </button>`;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── HTTP Helpers ──────────────────────────────────────
async function postJSON(url, data = {}) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify(data)
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error(err.error || `Request failed: ${res.status}`);
  }
  return res.json();
}

async function getJSON(url) {
  const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    throw new Error(err.error || `Request failed: ${res.status}`);
  }
  return res.json();
}

// ── Number Formatters ─────────────────────────────────
function formatNumber(n) {
  if (n == null) return '0';
  return Number(n).toLocaleString('id-ID');
}

function formatCurrency(n) {
  if (n == null) return 'Rp 0';
  return 'Rp ' + Number(n).toLocaleString('id-ID');
}

function formatPercent(n, decimals = 1) {
  if (n == null) return '0%';
  return Number(n).toFixed(decimals) + '%';
}

function formatDuration(hours) {
  if (hours == null) return '0j 0m';
  const h = Math.floor(hours);
  const m = Math.round((hours - h) * 60);
  return `${h}j ${m}m`;
}

// ── Skeleton Loaders ──────────────────────────────────
function showSkeleton(elementId) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.dataset.originalContent = el.innerHTML;
  el.innerHTML = `
    <div class="skeleton skeleton-line w-80"></div>
    <div class="skeleton skeleton-line w-60"></div>
    <div class="skeleton skeleton-line w-40"></div>`;
}

function hideSkeleton(elementId) {
  const el = document.getElementById(elementId);
  if (!el) return;
  if (el.dataset.originalContent) {
    el.innerHTML = el.dataset.originalContent;
    delete el.dataset.originalContent;
  }
}

function showSpinner(buttonEl) {
  if (!buttonEl) return;
  buttonEl.disabled = true;
  buttonEl.dataset.originalText = buttonEl.innerHTML;
  buttonEl.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status"></span>Processing...`;
}

function hideSpinner(buttonEl) {
  if (!buttonEl) return;
  buttonEl.disabled = false;
  if (buttonEl.dataset.originalText) {
    buttonEl.innerHTML = buttonEl.dataset.originalText;
    delete buttonEl.dataset.originalText;
  }
}

// ── Sidebar Mobile Toggle ─────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const toggleBtn = document.getElementById('sidebarToggle');
  const sidebar = document.getElementById('mainSidebar');

  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });

    // Close sidebar on outside click
    document.addEventListener('click', (e) => {
      if (!sidebar.contains(e.target) && !toggleBtn.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }
});

// ── Expose globally ───────────────────────────────────
window.showToast    = showToast;
window.postJSON     = postJSON;
window.getJSON      = getJSON;
window.formatNumber = formatNumber;
window.formatCurrency = formatCurrency;
window.formatPercent  = formatPercent;
window.formatDuration = formatDuration;
window.showSkeleton = showSkeleton;
window.hideSkeleton = hideSkeleton;
window.showSpinner  = showSpinner;
window.hideSpinner  = hideSpinner;

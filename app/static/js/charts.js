/* ====================================================
   StreamAnalytica — charts.js
   Premium Plotly.js Chart Builders
   ==================================================== */

'use strict';

// ── Base Layout Template ─────────────────────────────
const BASE_LAYOUT = {
  font: { family: "'Inter', system-ui, sans-serif", size: 12.5, color: '#0F172A' },
  plot_bgcolor: '#FFFFFF',
  paper_bgcolor: '#FFFFFF',
  margin: { l: 55, r: 30, t: 50, b: 50, pad: 4 },
  showlegend: true,
  legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1 },
  hoverlabel: {
    bgcolor: 'white',
    bordercolor: '#E2E8F0',
    font: { family: "'Inter', sans-serif", size: 12, color: '#0F172A' }
  },
  hovermode: 'closest',
};

const BASE_AXIS = {
  gridcolor: '#F1F5F9',
  zerolinecolor: '#E2E8F0',
  linecolor: '#E2E8F0',
  tickfont: { size: 11 },
};

const CONFIG = {
  displayModeBar: true,
  modeBarButtonsToRemove: ['toImage2', 'sendDataToCloud'],
  responsive: true,
  displaylogo: false,
};

const COLORS = {
  primary: '#2563EB',
  success: '#10B981',
  warning: '#F59E0B',
  danger:  '#EF4444',
  info:    '#6366F1',
  orange:  '#F97316',
  surface: 'rgba(37, 99, 235, 0.65)',
  scatter: '#EF4444',
};

// ── 1. Actual vs Predicted Scatter ───────────────────
function renderActualVsPredicted(elementId, actualData, predictedData) {
  const minVal = Math.min(...actualData, ...predictedData);
  const maxVal = Math.max(...actualData, ...predictedData);
  const pad    = (maxVal - minVal) * 0.05;

  const traces = [
    {
      x: predictedData, y: actualData,
      mode: 'markers',
      type: 'scatter',
      name: 'Observations',
      marker: { color: COLORS.primary, size: 7, opacity: 0.75,
                line: { color: 'white', width: 1 } },
    },
    {
      x: [minVal - pad, maxVal + pad],
      y: [minVal - pad, maxVal + pad],
      mode: 'lines', type: 'scatter',
      name: 'Perfect Fit',
      line: { color: COLORS.danger, width: 1.5, dash: 'dash' },
    }
  ];

  const layout = {
    ...BASE_LAYOUT,
    title: { text: 'Actual vs. Predicted Sales', font: { size: 14, weight: 600 } },
    xaxis: { ...BASE_AXIS, title: 'Predicted Products Sold' },
    yaxis: { ...BASE_AXIS, title: 'Actual Products Sold' },
  };

  Plotly.react(elementId, traces, layout, CONFIG);
}

// ── 2. Residual Plot ─────────────────────────────────
function renderResidualPlot(elementId, predictedData, residuals) {
  const traces = [
    {
      x: predictedData, y: residuals,
      mode: 'markers', type: 'scatter',
      name: 'Residuals',
      marker: {
        color: residuals.map(r => r > 0 ? COLORS.success : COLORS.danger),
        size: 7, opacity: 0.75, line: { color: 'white', width: 1 }
      },
    },
    {
      x: [Math.min(...predictedData), Math.max(...predictedData)],
      y: [0, 0],
      mode: 'lines', type: 'scatter',
      name: 'Zero Line',
      line: { color: COLORS.warning, width: 1.5, dash: 'dot' },
    }
  ];

  const layout = {
    ...BASE_LAYOUT,
    title: { text: 'Residual Analysis', font: { size: 14, weight: 600 } },
    xaxis: { ...BASE_AXIS, title: 'Predicted Value' },
    yaxis: { ...BASE_AXIS, title: 'Residual (Actual − Predicted)' },
  };

  Plotly.react(elementId, traces, layout, CONFIG);
}

// ── 3. 3D Surface ────────────────────────────────────
function renderSurface3D(elementId, durGrid, viewGrid, salesMesh, actualPoints) {
  const traces = [
    {
      type: 'surface', x: durGrid, y: viewGrid, z: salesMesh,
      name: 'Regression Surface',
      colorscale: [
        [0, '#EFF6FF'], [0.25, '#93C5FD'], [0.5, '#2563EB'],
        [0.75, '#1D4ED8'], [1, '#1E3A8A']
      ],
      opacity: 0.82,
      contours: { z: { show: true, project: { z: true }, color: 'white', width: 1 } },
      showscale: true,
      colorbar: { title: 'Sales', thickness: 12, len: 0.6 },
    }
  ];

  if (actualPoints && actualPoints.x && actualPoints.x.length > 0) {
    traces.push({
      type: 'scatter3d',
      x: actualPoints.x, y: actualPoints.y, z: actualPoints.z,
      mode: 'markers', name: 'Actual Sessions',
      marker: { color: COLORS.danger, size: 3.5, opacity: 0.8,
                line: { color: 'white', width: 0.5 } },
    });
  }

  const layout = {
    ...BASE_LAYOUT,
    title: { text: '3D Regression Surface', font: { size: 14, weight: 600 } },
    scene: {
      xaxis: { title: 'Duration (h)', gridcolor: '#E2E8F0' },
      yaxis: { title: 'Active Viewers', gridcolor: '#E2E8F0' },
      zaxis: { title: 'Products Sold', gridcolor: '#E2E8F0' },
      camera: { eye: { x: 1.6, y: 1.4, z: 1.0 } },
    },
    margin: { l: 0, r: 0, t: 50, b: 0 },
    legend: { x: 0.01, y: 0.99 },
  };

  Plotly.react(elementId, traces, layout, CONFIG);
}

// ── 4. Loss Curve ────────────────────────────────────
function renderLossCurve(elementId, epochs, losses) {
  const traces = [{
    x: epochs, y: losses,
    type: 'scatter', mode: 'lines',
    name: 'MSE Loss',
    line: { color: COLORS.primary, width: 2.5, shape: 'spline' },
    fill: 'tozeroy', fillcolor: 'rgba(37, 99, 235, 0.08)',
  }];

  const layout = {
    ...BASE_LAYOUT,
    title: { text: 'Training Loss Curve', font: { size: 13, weight: 600 } },
    xaxis: { ...BASE_AXIS, title: 'Epoch' },
    yaxis: { ...BASE_AXIS, title: 'MSE Loss (Scaled)' },
    margin: { l: 55, r: 20, t: 45, b: 45 },
  };

  Plotly.react(elementId, traces, layout, CONFIG);
}

// ── 5. Seasonality Bar ───────────────────────────────
function renderSeasonalityBar(elementId, months, salesValues, viewerValues) {
  const traces = [
    {
      x: months, y: salesValues,
      type: 'bar', name: 'Avg. Products Sold',
      marker: {
        color: salesValues.map((v, i) => {
          const max = Math.max(...salesValues);
          return i === salesValues.indexOf(max) ? COLORS.primary : 'rgba(37, 99, 235, 0.45)';
        }),
        borderradius: 4,
      },
    }
  ];

  if (viewerValues && viewerValues.length > 0) {
    traces.push({
      x: months, y: viewerValues,
      type: 'scatter', mode: 'lines+markers', name: 'Avg. Viewers',
      yaxis: 'y2',
      line: { color: COLORS.warning, width: 2 },
      marker: { color: COLORS.warning, size: 6 },
    });
  }

  const layout = {
    ...BASE_LAYOUT,
    title: { text: 'Monthly Sales Seasonality', font: { size: 14, weight: 600 } },
    xaxis: { ...BASE_AXIS, title: '' },
    yaxis: { ...BASE_AXIS, title: 'Avg. Products Sold' },
    yaxis2: viewerValues ? {
      ...BASE_AXIS, title: 'Avg. Viewers',
      overlaying: 'y', side: 'right', showgrid: false
    } : undefined,
    barmode: 'group',
  };

  Plotly.react(elementId, traces, layout, CONFIG);
}

// ── 6. Moving Average ────────────────────────────────
function renderMovingAverage(elementId, actual, maValues, labels) {
  const x = labels || actual.map((_, i) => i + 1);
  const traces = [
    {
      x, y: actual, type: 'scatter', mode: 'lines',
      name: 'Actual Sales', opacity: 0.45,
      line: { color: COLORS.info, width: 1.5 },
    },
    {
      x, y: maValues, type: 'scatter', mode: 'lines',
      name: '7-Session Moving Avg',
      line: { color: COLORS.primary, width: 2.5, shape: 'spline' },
    }
  ];

  const layout = {
    ...BASE_LAYOUT,
    title: { text: 'Sales Moving Average Trend', font: { size: 14, weight: 600 } },
    xaxis: { ...BASE_AXIS, title: 'Session #' },
    yaxis: { ...BASE_AXIS, title: 'Products Sold' },
  };

  Plotly.react(elementId, traces, layout, CONFIG);
}

// ── 7. Correlation Heatmap ───────────────────────────
function renderCorrelationHeatmap(elementId, labels, matrix) {
  const cleanLabels = labels.map(l =>
    l === 'Durasi_Jam' ? 'Duration (h)' :
    l === 'Penonton Aktif' ? 'Active Viewers' :
    l === 'Produk Terjual' ? 'Products Sold' : l
  );

  const traces = [{
    type: 'heatmap', z: matrix, x: cleanLabels, y: cleanLabels,
    colorscale: [
      [0, '#FEF2F2'], [0.25, '#FCA5A5'], [0.5, '#F8FAFC'],
      [0.75, '#93C5FD'], [1, '#1D4ED8']
    ],
    zmin: -1, zmax: 1,
    text: matrix.map(row => row.map(v => v.toFixed(3))),
    texttemplate: '%{text}',
    showscale: true,
    colorbar: { thickness: 12, len: 0.8, title: 'r' },
  }];

  const layout = {
    ...BASE_LAYOUT,
    title: { text: 'Feature Correlation Matrix', font: { size: 14, weight: 600 } },
    xaxis: { ...BASE_AXIS, side: 'bottom' },
    yaxis: { ...BASE_AXIS, autorange: 'reversed' },
    margin: { l: 120, r: 60, t: 50, b: 80 },
  };

  Plotly.react(elementId, traces, layout, CONFIG);
}

// ── 8. Sales Trend ───────────────────────────────────
function renderSalesTrend(elementId, sessionNums, salesValues, maValues) {
  const traces = [
    {
      x: sessionNums, y: salesValues,
      type: 'bar', name: 'Products Sold',
      marker: { color: 'rgba(37, 99, 235, 0.4)', borderradius: 3 },
    },
    {
      x: sessionNums, y: maValues,
      type: 'scatter', mode: 'lines', name: '7-Session MA',
      line: { color: COLORS.primary, width: 2.5, shape: 'spline' },
    }
  ];

  const layout = {
    ...BASE_LAYOUT,
    title: { text: 'Sales Trend', font: { size: 14, weight: 600 } },
    xaxis: { ...BASE_AXIS, title: 'Session #' },
    yaxis: { ...BASE_AXIS, title: 'Products Sold' },
    barmode: 'overlay',
  };

  Plotly.react(elementId, traces, layout, CONFIG);
}

// ── 9. Model Comparison Bar ──────────────────────────
function renderModelComparisonBar(elementId, modelNames, metricValues, metricName, colorHigh) {
  const maxVal = Math.max(...metricValues);
  const traces = [{
    x: modelNames, y: metricValues,
    type: 'bar', name: metricName,
    marker: {
      color: metricValues.map(v => v === maxVal ? COLORS.primary : 'rgba(37,99,235,0.35)'),
      borderradius: 6,
    },
    text: metricValues.map(v => v.toFixed(4)),
    textposition: 'auto',
  }];

  const layout = {
    ...BASE_LAYOUT,
    title: { text: metricName + ' Comparison', font: { size: 14, weight: 600 } },
    xaxis: { ...BASE_AXIS },
    yaxis: { ...BASE_AXIS, title: metricName },
    showlegend: false,
    margin: { l: 55, r: 20, t: 50, b: 60 },
  };

  Plotly.react(elementId, traces, layout, CONFIG);
}

// ── Expose globally ───────────────────────────────────
window.renderActualVsPredicted  = renderActualVsPredicted;
window.renderResidualPlot       = renderResidualPlot;
window.renderSurface3D          = renderSurface3D;
window.renderLossCurve          = renderLossCurve;
window.renderSeasonalityBar     = renderSeasonalityBar;
window.renderMovingAverage      = renderMovingAverage;
window.renderCorrelationHeatmap = renderCorrelationHeatmap;
window.renderSalesTrend         = renderSalesTrend;
window.renderModelComparisonBar = renderModelComparisonBar;

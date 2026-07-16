# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-07-16

### Added
- **Flask MVC Port**: Full migration of the Streamlit student dashboard into a professional Flask server architecture with standard Blueprint routing.
- **Constrained SGD Engine**: Background training processor using custom Stochastic Gradient Descent (SGD) enforcing non-negative weights ($\theta \ge 0, bias \ge 0$) matching the core thesis specifications.
- **Server-Sent Events (SSE)**: Built-in streaming API endpoint (`/api/v1/training/stream`) to deliver real-time loss, RMSE, delta, and ETA convergence updates directly to browser consoles.
- **Interactive Plotly.js Visualizations**: Premium white-themed responsive visual charts including a rotating 3D regression surface mesh, Actual vs. Predicted scatter, residuals analysis, moving average trendlines, and grouped seasonality bars.
- **Azure/Copilot Style Prediction Center**: Redesigned Prediction Center with simulated test presets (High-Traffic Peak, Standard broadcast, Marathon Run), a multi-step checklist execution overlay, and scroll-locked side-by-side configuration.
- **Tableau-Style Executive PDF Preview**: Custom letter-sized PDF preview sheet rendering in the Reports page with full-featured toolbar allowing Print and Open in New Window actions.
- **Clean Model Registry & Export Exporters**: Active model activation controls, registry purge controls, and automated download step animations (Excel, PDF, PNG, Markdown).

### Fixed
- **Reports NoneType Bug**: Caught `AttributeError` exceptions inside the reports API when no active models are configured, showing a clean fallback layout with direct link to the Training Center.
- **Missing Tabulate Dependency**: Removed dependency on `tabulate` inside the markdown summary generation, implementing manual conversion for baseline statistics.

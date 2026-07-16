"""Reports API — PDF, Excel, PNG, Markdown downloads"""
import io
from flask import Blueprint, jsonify, current_app, send_file, Response
from src.services.reporting_service import ReportingService
from src.repositories.model_repository import ModelRepository

reports_api_bp = Blueprint('reports_api', __name__, url_prefix='/reports')


def _get_active_model():
    try:
        repo = ModelRepository(registry_dir=current_app.config['MODEL_DIR'])
        return repo.get_active_model()
    except Exception:
        return None


@reports_api_bp.route('/excel', methods=['GET'])
def download_excel():
    try:
        df = current_app.config.get('DATASET')
        model_data = _get_active_model()
        if model_data is None:
            return jsonify({
                'error': 'NO_MODEL',
                'message': 'No trained model available. Please train a model first.'
            }), 400

        rs = ReportingService()
        excel_bytes = rs.generate_excel_report(df, model_data)
        return send_file(
            io.BytesIO(excel_bytes),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='streamanalytica_report.xlsx'
        )
    except Exception as e:
        return jsonify({'error': 'SYSTEM_ERROR', 'message': str(e)}), 500


@reports_api_bp.route('/png', methods=['GET'])
def download_png():
    try:
        df = current_app.config.get('DATASET')
        model_data = _get_active_model()
        if model_data is None:
            return jsonify({
                'error': 'NO_MODEL',
                'message': 'No trained model available. Please train a model first.'
            }), 400

        rs = ReportingService()
        png_bytes = rs.generate_chart_png(df, model_data)
        return send_file(
            io.BytesIO(png_bytes),
            mimetype='image/png',
            as_attachment=True,
            download_name='streamanalytica_chart.png'
        )
    except Exception as e:
        return jsonify({'error': 'SYSTEM_ERROR', 'message': str(e)}), 500


@reports_api_bp.route('/markdown', methods=['GET'])
def download_markdown():
    try:
        df = current_app.config.get('DATASET')
        model_data = _get_active_model()
        if model_data is None:
            return jsonify({
                'error': 'NO_MODEL',
                'message': 'No trained model available. Please train a model first.'
            }), 400

        rs = ReportingService()
        md_text = rs.generate_markdown_report(df, model_data)
        return Response(
            md_text,
            mimetype='text/markdown',
            headers={'Content-Disposition': 'attachment; filename=streamanalytica_report.md'}
        )
    except Exception as e:
        return jsonify({'error': 'SYSTEM_ERROR', 'message': str(e)}), 500


@reports_api_bp.route('/pdf', methods=['GET'])
def download_pdf():
    try:
        df = current_app.config.get('DATASET')
        model_data = _get_active_model()
        if model_data is None:
            return jsonify({
                'error': 'NO_MODEL',
                'message': 'No trained model available. Please train a model first.'
            }), 400

        rs = ReportingService()
        try:
            pdf_bytes = rs.generate_pdf_report(df, model_data)
            return send_file(
                io.BytesIO(pdf_bytes),
                mimetype='application/pdf',
                as_attachment=True,
                download_name='streamanalytica_report.pdf'
            )
        except AttributeError:
            excel_bytes = rs.generate_excel_report(df, model_data)
            return send_file(
                io.BytesIO(excel_bytes),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='streamanalytica_report.xlsx'
            )
    except Exception as e:
        return jsonify({'error': 'SYSTEM_ERROR', 'message': str(e)}), 500


@reports_api_bp.route('/preview', methods=['GET'])
def preview_markdown():
    """Returns markdown report text for preview panel."""
    try:
        df = current_app.config.get('DATASET')
        model_data = _get_active_model()
        if model_data is None:
            return jsonify({
                'error': 'NO_MODEL',
                'message': 'No trained model available. Please train a model first.'
            }), 400

        rs = ReportingService()
        md_text = rs.generate_markdown_report(df, model_data)
        return jsonify({'markdown': md_text})
    except Exception as e:
        return jsonify({
            'error': 'SYSTEM_ERROR',
            'message': 'An internal system error occurred while generating report preview.',
            'details': str(e)
        }), 500

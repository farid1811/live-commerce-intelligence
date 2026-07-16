"""StreamAnalytica - Flask Application Entry Point
Run with: python run.py
Access at: http://localhost:5000
"""
import sys
import os

# Ensure project root is first in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

app = create_app()

if __name__ == '__main__':
    print("=" * 60)
    print("  StreamAnalytica - AI Live Commerce Intelligence")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

import os

class Config:
    """StreamAnalytica application configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'streamanalytica-secret-2024-enterprise')
    
    # Base paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Dataset
    DATASET_PATH = os.path.join(BASE_DIR, 'Data Skripsi full.xlsx')
    DATASET_SHEET = 'Sheet1'
    DATASET_HEADER_ROW = 1
    
    # Storage
    MODEL_DIR = os.path.join(BASE_DIR, 'storage', 'models_registry')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'storage', 'uploads')
    
    # Flask
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
    
    # Training defaults
    DEFAULT_LEARNING_RATE = 0.01
    DEFAULT_MAX_EPOCHS = 1000
    DEFAULT_TOLERANCE = 0.001
    
    # Product info
    PRODUCT_NAME = 'StreamAnalytica'
    PRODUCT_TAGLINE = 'Predict. Optimize. Dominate Live Commerce.'
    PRODUCT_VERSION = '1.0.0'


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

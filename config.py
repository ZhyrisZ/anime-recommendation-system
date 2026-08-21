import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    DEBUG = False
    TESTING = False
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///anime_recommendations.db')
    
    # Recommendation settings
    MIN_RATINGS = 3  # Minimum ratings for a user to get recommendations
    TOP_N_RECOMMENDATIONS = 10
    SIMILARITY_THRESHOLD = 0.5
    
    # Algorithm settings
    COLLABORATIVE_N_NEIGHBORS = 5
    CONTENT_WEIGHT = 0.4
    COLLABORATIVE_WEIGHT = 0.6

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DATABASE_URL = 'sqlite:///test_anime.db'

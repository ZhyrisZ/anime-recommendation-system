"""Main application entry point."""

import logging
from src.api.routes import create_app
from config import DevelopmentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    app = create_app()
    app.config.from_object(DevelopmentConfig)
    app.run(debug=True, host='0.0.0.0', port=5000)

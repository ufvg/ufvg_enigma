# utils/logging_config.py
import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir='logs'):
    """
    Configure comprehensive logging
    """
    # Create logs directory if not exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            RotatingFileHandler(
                os.path.join(log_dir, 'trading_bot.log'),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )

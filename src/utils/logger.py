"""
Logging utilities
Hem console hem file logging desteği
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


class Logger:
    """Custom logger with file and console handlers"""
    
    def __init__(self, name: str, config: dict):
        """
        Args:
            name: Logger ismi (genelde __name__)
            config: logging config dict
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config['level']))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if config.get('log_to_console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if config.get('log_to_file', True):
            log_dir = Path(config['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def critical(self, msg):
        self.logger.critical(msg)


def get_logger(name: str, config: dict) -> Logger:
    """Logger factory"""
    return Logger(name, config)
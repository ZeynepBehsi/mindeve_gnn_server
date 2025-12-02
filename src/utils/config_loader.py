"""
YAML config loader with environment detection
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import socket


class ConfigLoader:
    """Load and merge YAML configs"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.is_server = self._detect_environment()
    
    def _detect_environment(self) -> bool:
        """
        Server mı local mi tespit et
        Server hostname'inde genelde 'server' veya 'gpu' var
        """
        hostname = socket.gethostname().lower()
        return any(x in hostname for x in ['server', 'gpu', 'cuda', 'rtx'])
    
    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Config dosyasını yükle
        
        Args:
            config_name: 'base', 'clustering', 'gnn'
        
        Returns:
            Config dictionary
        """
        config_path = self.config_dir / f"{config_name}_config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Environment-specific overrides
        if config_name == 'base' and 'data' in config:
            if self.is_server:
                print("🖥️  Server environment detected")
                config['data'].update(config['data']['server'])
            else:
                print("💻 Local environment detected")
                config['data'].update(config['data']['local'])
        
        return config
    
    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Tüm config'leri yükle"""
        return {
            'base': self.load('base'),
            'clustering': self.load('clustering'),
            'gnn': self.load('gnn')
        }
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Config'leri merge et"""
        merged = {}
        for config in configs:
            merged.update(config)
        return merged


# Quick loader function
def load_config(config_name: str = 'base') -> Dict[str, Any]:
    """Shortcut function"""
    loader = ConfigLoader()
    return loader.load(config_name)
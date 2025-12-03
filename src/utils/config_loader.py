"""
Configuration loader with environment detection and deep merge
"""

import yaml
from pathlib import Path
import socket


class ConfigLoader:
    """Load YAML configs with environment detection"""
    
    def __init__(self, config_dir: str = 'config'):
        self.config_dir = Path(config_dir)
        
        # Detect environment
        hostname = socket.gethostname().lower()
        if 'server' in hostname or 'gpu' in hostname or 'rtx' in hostname:
            self.env = 'server'
        else:
            self.env = 'local'
        
        print(f"💻 {self.env.title()} environment detected")
    
    def load(self, config_name: str) -> dict:
        """
        Load a single config file
        
        Args:
            config_name: Name of config (without '_config.yaml')
        
        Returns:
            Dictionary with config
        """
        config_path = self.config_dir / f'{config_name}_config.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Environment-specific override
        env_key = f"{self.env}_config"
        if env_key in config:
            config = self._deep_merge(config, config[env_key])
            del config[env_key]
        
        return config
    
    def load_all(self) -> dict:
        """
        Load and deep merge all configs
        
        Returns:
            Dictionary with merged configs from base, clustering, gnn
        """
        configs = ['base', 'clustering', 'gnn']
        merged = {}
        
        for config_name in configs:
            try:
                config = self.load(config_name)
                merged = self._deep_merge(merged, config)
            except FileNotFoundError:
                print(f"⚠️  Warning: {config_name}_config.yaml not found, skipping")
        
        return merged
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """
        Deep merge two dictionaries
        
        Args:
            base: Base dictionary
            override: Dictionary to merge into base
        
        Returns:
            Merged dictionary
        
        Example:
            base = {'a': {'b': 1, 'c': 2}}
            override = {'a': {'c': 3, 'd': 4}}
            result = {'a': {'b': 1, 'c': 3, 'd': 4}}
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = value
        
        return result


def load_config(config_name: str) -> dict:
    """
    Quick loader function for single config
    
    Args:
        config_name: Name of config file
    
    Returns:
        Config dictionary
    
    Example:
        config = load_config('base')
    """
    loader = ConfigLoader()
    return loader.load(config_name)


def load_all_configs() -> dict:
    """
    Quick loader function for all configs
    
    Returns:
        Merged config dictionary
    
    Example:
        config = load_all_configs()
    """
    loader = ConfigLoader()
    return loader.load_all()
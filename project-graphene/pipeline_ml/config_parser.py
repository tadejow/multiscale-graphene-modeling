"""
Module for parsing YAML configuration files.
Complies with Single Responsibility Principle (SRP).
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any


class ConfigParser:
    """Class responsible for loading and parsing YAML configuration files."""

    def __init__(self, config_path: str | Path):
        """
        Initializes the ConfigParser.

        Args:
            config_path (str | Path): Path to the YAML configuration file.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

    def load(self) -> Dict[str, Any]:
        """
        Loads the YAML file into a Python dictionary.

        Returns:
            Dict[str, Any]: Parsed configuration parameters.
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
                logging.info(f"Successfully loaded configuration from {self.config_path}")
                return config
            except yaml.YAMLError as e:
                logging.error(f"Error parsing YAML file: {e}")
                raise

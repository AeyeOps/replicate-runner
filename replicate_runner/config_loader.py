import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

class ConfigLoader:
    def __init__(self):
        """
        Initializes the config loader by:
          - Loading environment from `.env`
          - Merging YAML configs from `config/` folder
        """
        self._load_env()
        self.config = self._load_yaml_configs()

    def _load_env(self):
        """
        Load environment variables from `.env` in the current working directory
        or up to two parents up.
        """
        cwd = Path.cwd()
        possible_locations = [
            cwd / ".env",
            cwd.parent / ".env",
            cwd.parent.parent / ".env",
        ]
        for location in possible_locations:
            if location.exists():
                load_dotenv(location, override=True)
                break

    def _load_yaml_configs(self):
        """
        Load all YAML files from `config` directory (under this project).
        Merge them if necessary.
        """
        config_dir = Path(__file__).parent.parent / "config"
        merged_config = {}
        for yaml_file in config_dir.glob("*.yaml"):
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                merged_config.update(data)
        return merged_config

    def get(self, key, default=None):
        """
        Retrieve config item from merged config or environment variables.
        Environment variables take precedence if you want them to override.
        """
        return os.environ.get(key, self.config.get(key, default))

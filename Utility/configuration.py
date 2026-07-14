from __future__ import annotations

import logging
import os
from configparser import ExtendedInterpolation, RawConfigParser
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "config.ini"


class Config:
    """Typed access to the application's INI configuration."""

    def __init__(self, config_filename: os.PathLike | str | None = None):
        configured = config_filename or os.environ.get("ENERGY_COMMUNITY_CONFIG")
        self.config_filename = Path(configured or DEFAULT_CONFIG_FILE).expanduser().resolve()
        self.project_root = PROJECT_ROOT
        self.__config = RawConfigParser(
            allow_no_value=True,
            interpolation=ExtendedInterpolation(),
        )
        if not self.config_filename.is_file():
            raise FileNotFoundError(f"Configuration file not found: {self.config_filename}")
        with self.config_filename.open(encoding="utf-8") as stream:
            self.__config.read_file(stream)
        self._registered_entries = {
            "data": {
                "tilt_range": self._get_tilt_range,
                "azimuth_range": self._get_azimuth_range,
            }
        }

    def get(self, section, key, fallback=None):
        if section not in self._registered_entries or key not in self._registered_entries[section]:
            return self._get(section, key, fallback)
        return self._registered_entries[section][key](section, key, fallback)

    def getstring(self, section, key, fallback=None):
        value = self.__config.get(section, key, fallback=fallback)
        if value is None:
            raise KeyError(f"Section '{section}', key '{key}' not found in configuration")
        return value

    def getpath(self, section, key, fallback=None, *, resolve: bool = True) -> Path:
        """Return a configured path, resolving relative paths from the project root."""
        raw = os.path.expandvars(os.path.expanduser(self.getstring(section, key, fallback)))
        path = Path(raw)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve() if resolve else path

    def __get_np_arange(self, section, key, fallback):
        return np.arange(*self.getarray(section, key, dtype=int, fallback=fallback))

    def _get_tilt_range(self, section, key, fallback):
        return self.__get_np_arange(section, key, fallback)

    def _get_azimuth_range(self, section, key, fallback):
        return self.__get_np_arange(section, key, fallback)

    def _get(self, section, key, fallback=None):
        try:
            value = self.__config.get(section, key, fallback=fallback)
        except Exception as exc:
            raise KeyError(
                f"Section '{section}', key '{key}' problem in configuration: '{exc}'"
            ) from exc
        if value is None:
            raise KeyError(f"Section '{section}', key '{key}' not found in configuration")
        if "," not in value:
            return value
        values = list(filter(len, value.strip("][").split(",")))
        if all(element.strip().lstrip("-").isdigit() for element in values):
            return [int(x) for x in values]
        return [value.strip() for value in values]

    def set(self, section, key, value):
        self.__config.set(section, key, str(value))

    def setboolean(self, section, key, value):
        self.__config.set(section, key, "True" if value else "False")

    def getboolean(self, section, key, fallback=None):
        return self.__config.getboolean(section, key, fallback=fallback)

    def getint(self, section, key, fallback=None):
        return self.__config.getint(section, key, fallback=fallback)

    def getarray(self, section, key, dtype=str, fallback=None):
        value = self._get(section, key, fallback=fallback)
        if isinstance(value, str):
            value = value.strip().split(",")
        try:
            return [dtype(item) for item in value]
        except TypeError:
            return [dtype(value)]

    def getfloat(self, section, key, fallback=None):
        return self.__config.getfloat(section, key, fallback=fallback)

    def has_option(self, section, option):
        return self.__config.has_option(section, option)

    def has_section(self, section):
        return self.__config.has_section(section)


# Backward-compatible name for older imports.
ConfigurationManager = Config

# The single shared application configuration instance.
config = Config()

import logging
from configparser import RawConfigParser, ExtendedInterpolation
from os import getcwd
from os.path import join, dirname
from sys import argv

import numpy as np

logger = logging.getLogger(__name__)


class ConfigurationManager:
    def __init__(self, config_filename=join(getcwd(), "..", "config", "config/config.ini")):
        self.__config = RawConfigParser(allow_no_value=True, interpolation=ExtendedInterpolation())
        self.__config.read_file(open(config_filename))
        self._registered_entries = {"data": {"tilt_range": self._get_tilt_range,
                                             "azimuth_range": self._get_azimuth_range}}

    def get(self, section, key, fallback=None):
        if section not in self._registered_entries or key not in self._registered_entries[section]:
            return self._get(section, key, fallback)
        return self._registered_entries[section][key](section, key, fallback)

    def getstring(self, section, key, fallback=None):
        return self.__config.get(section, key, fallback=fallback)

    def __get_np_arange(self, section, key, fallback):
        return np.arange(*self.getarray(section, key, dtype=int, fallback=fallback))

    def _get_tilt_range(self, section, key, fallback):
        return self.__get_np_arange(section, key, fallback)

    def _get_azimuth_range(self, section, key, fallback):
        return self.__get_np_arange(section, key, fallback)

    def _get(self, section, key, fallback=None):
        try:
            value = self.__config.get(section, key, fallback=fallback)
        except Exception as e:
            raise KeyError(f"Section '{section}', key '{key}' problem in configuration: '{e}'")
        if value is None:
            raise KeyError(f"Section '{section}', key '{key}' not found in configuration")

        if "," not in value:
            return value

        values = list(filter(len, value.strip('][').split(',')))
        all_integers = all(element.isdigit() for element in values)
        if all_integers:
            return [int(x) for x in values]

        return [v.replace(" ", "") for v in values]

    def set(self, section, key, value):
        self.__config.set(section, key, value)

    def setboolean(self, section, key, value):
        boolean_str = 'True' if value else 'False'
        self.__config.set(section, key, boolean_str)

    def getboolean(self, section, key, fallback=None):
        return self.__config.getboolean(section, key, fallback=fallback)

    def getint(self, section, key, fallback=None):
        try:
            return self.__config.getint(section, key, fallback=fallback)
        except:
            return self.__config.get(section, key, fallback=fallback)

    def getarray(self, section, key, dtype=str,fallback=None):
        val = self._get(section, key, fallback=fallback)
        if type(val) == str:
            val = val.strip().split(",")
        try:
            return [dtype(v) for v in val]
        except TypeError:
            return [dtype(val), ]

    def getfloat(self, section, key, fallback=None):
        return self.__config.getfloat(section, key, fallback=fallback)

    def has_option(self, section, option):
        return self.__config.has_option(section, option)


# def do_configuration():
config_file = argv[2] if len(argv) >= 3 else join(dirname(__file__), "..", 'config', 'config.ini')
config = ConfigurationManager(config_filename=config_file)

from configparser import ConfigParser
import os

CONFIG_FILE_PATH = 'config.ini'

def get_config_file():
    return os.environ.get('CONFIG_FILE', CONFIG_FILE_PATH)

CONFIG_FILE = get_config_file()

def create_config(config_file=None):
    config = ConfigParser()
    config.read(config_file or CONFIG_FILE)
    return config

CONFIG = create_config()
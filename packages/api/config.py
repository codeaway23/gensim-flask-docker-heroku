import os
import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

class Config:
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    # SERVER_PORT = 8000


class ProductionConfig(Config):
    DEBUG = False
    # SERVER_PORT = os.environ.get('PORT', 8000)


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    DEBUG = True

config_by_name = dict(
    dev=DevelopmentConfig,
    test=TestingConfig,
    prod=ProductionConfig
    )
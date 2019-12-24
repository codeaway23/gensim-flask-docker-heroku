from api.controller import prediction_app
from api.config import config_by_name
from flask import Flask

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])
    app.register_blueprint(prediction_app)
    return app
from flask import Flask
from flask_cors import CORS

from app.config import config
from app.database import init_db


def create_app(config_name: str = "default") -> Flask:
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    init_db(app)

    from app.api.routes import bp as api_bp
    app.register_blueprint(api_bp)

    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Not found"}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {"error": "Internal server error"}, 500

    return app

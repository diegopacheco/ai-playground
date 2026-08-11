"""
Database module.
Provides SQLAlchemy instance and database initialization utilities.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db(app):
    """Initialize the database with the Flask app context."""
    db.init_app(app)

    with app.app_context():
        # Import models so they are registered with SQLAlchemy
        from app.models.product import Product  # noqa: F401

        db.create_all()

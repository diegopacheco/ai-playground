"""
Product data model.
Defines the Product entity and its database schema.
"""

from datetime import datetime

from app.database import db


class Product(db.Model):
    """
    Product model representing an item in the catalog.

    Attributes:
        id: Unique identifier (auto-incremented).
        name: Product name (required, max 200 chars).
        description: Product description (optional, max 1000 chars).
        price: Product price in USD (required, must be >= 0).
        category: Product category (required, max 100 chars).
        in_stock: Whether the product is in stock (default True).
        created_at: Timestamp of creation.
        updated_at: Timestamp of last update.
    """

    __tablename__ = "products"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(200), nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)
    price = db.Column(db.Numeric(10, 2), nullable=False)
    category = db.Column(db.String(100), nullable=False, index=True)
    in_stock = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(
        db.DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def to_dict(self):
        """Serialize the product to a dictionary for JSON responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "price": float(self.price),
            "category": self.category,
            "in_stock": self.in_stock,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f"<Product {self.name}>"

"""
Product service module.
Encapsulates all business logic for product operations.
"""

from typing import Dict, List, Optional

from app.database import db
from app.models.product import Product


class ProductService:
    """
    Service class for managing product CRUD operations.

    All database interactions are encapsulated here, keeping the API
    layer thin and focused on HTTP concerns.
    """

    @staticmethod
    def get_all() -> List[Dict]:
        """Retrieve all products, ordered by creation date (newest first)."""
        products = Product.query.order_by(Product.created_at.desc()).all()
        return [p.to_dict() for p in products]

    @staticmethod
    def get_by_id(product_id: int) -> Optional[Dict]:
        """Retrieve a single product by its ID."""
        product = Product.query.get(product_id)
        return product.to_dict() if product else None

    @staticmethod
    def create(data: Dict) -> Dict:
        """
        Create a new product.

        Args:
            data: Dictionary with keys name, description, price, category, in_stock.

        Returns:
            The created product as a dictionary.

        Raises:
            ValueError: If validation fails.
        """
        ProductService._validate(data)

        product = Product(
            name=data["name"],
            description=data.get("description"),
            price=float(data["price"]),
            category=data["category"],
            in_stock=data.get("in_stock", True),
        )
        db.session.add(product)
        db.session.commit()
        return product.to_dict()

    @staticmethod
    def update(product_id: int, data: Dict) -> Optional[Dict]:
        """
        Update an existing product.

        Args:
            product_id: The ID of the product to update.
            data: Dictionary with fields to update.

        Returns:
            The updated product as a dictionary, or None if not found.

        Raises:
            ValueError: If validation fails.
        """
        ProductService._validate(data)

        product = Product.query.get(product_id)
        if not product:
            return None

        product.name = data["name"]
        product.description = data.get("description")
        product.price = float(data["price"])
        product.category = data["category"]
        product.in_stock = data.get("in_stock", True)

        db.session.commit()
        return product.to_dict()

    @staticmethod
    def delete(product_id: int) -> bool:
        """
        Delete a product by ID.

        Returns:
            True if the product was deleted, False if not found.
        """
        product = Product.query.get(product_id)
        if not product:
            return False

        db.session.delete(product)
        db.session.commit()
        return True

    @staticmethod
    def _validate(data: Dict) -> None:
        """Validate product data before create/update."""
        required_fields = ["name", "price", "category"]
        for field in required_fields:
            if field not in data or data[field] is None:
                raise ValueError(f"Field '{field}' is required")

        try:
            price = float(data["price"])
            if price < 0:
                raise ValueError("Price must be non-negative")
        except (TypeError, ValueError) as exc:
            if "Price must be" in str(exc):
                raise
            raise ValueError("Price must be a valid number")

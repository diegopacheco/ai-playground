"""
API schemas module.
Defines request/response validation schemas using plain Python dicts.
"""

from typing import Dict, List, Tuple


class ProductSchema:
    """
    Schema for validating product data in API requests.

    Uses a simple validation approach without external dependencies
    (no Marshmallow needed), keeping the backend lightweight.
    """

    REQUIRED_FIELDS: List[str] = ["name", "price", "category"]
    OPTIONAL_FIELDS: List[str] = ["description", "in_stock"]

    @classmethod
    def validate_create(cls, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate data for creating a product.

        Returns:
            A tuple of (is_valid, list_of_errors).
        """
        errors = []

        if not isinstance(data, dict):
            return False, ["Request body must be a JSON object"]

        for field in cls.REQUIRED_FIELDS:
            if field not in data or data[field] is None or data[field] == "":
                errors.append(f"Field '{field}' is required")

        # Validate price
        if "price" in data and data["price"] is not None:
            try:
                price = float(data["price"])
                if price < 0:
                    errors.append("Price must be non-negative")
            except (TypeError, ValueError):
                errors.append("Price must be a valid number")

        # Validate name length
        if "name" in data and data["name"]:
            if len(str(data["name"])) > 200:
                errors.append("Name must be 200 characters or fewer")

        # Validate category length
        if "category" in data and data["category"]:
            if len(str(data["category"])) > 100:
                errors.append("Category must be 100 characters or fewer")

        return len(errors) == 0, errors

    @classmethod
    def validate_update(cls, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate data for updating a product.
        Reuses create validation since all fields are needed for update.
        """
        return cls.validate_create(data)

    @classmethod
    def serialize(cls, product_dict: Dict) -> Dict:
        """Serialize a product dictionary for the response."""
        return product_dict

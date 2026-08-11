import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.database import db
from app.models.product import Product


def test_create_product():
    app = create_app()
    with app.app_context():
        db.drop_all()
        db.create_all()

        product = Product(
            name="Test Product",
            description="A test product",
            price=9.99,
            category="Test",
            in_stock=True,
        )
        db.session.add(product)
        db.session.commit()

        assert product.id is not None
        assert product.name == "Test Product"
        assert float(product.price) == 9.99
        print("test_create_product: PASSED")


def test_to_dict():
    app = create_app()
    with app.app_context():
        db.drop_all()
        db.create_all()

        product = Product(
            name="Dict Test",
            description="Testing to_dict",
            price=19.99,
            category="Test",
            in_stock=False,
        )
        db.session.add(product)
        db.session.commit()

        result = product.to_dict()
        assert result["name"] == "Dict Test"
        assert result["price"] == 19.99
        assert result["in_stock"] is False
        print("test_to_dict: PASSED")


def test_delete_product():
    app = create_app()
    with app.app_context():
        db.drop_all()
        db.create_all()

        product = Product(
            name="Delete Me",
            price=5.00,
            category="Test",
        )
        db.session.add(product)
        db.session.commit()
        pid = product.id

        db.session.delete(product)
        db.session.commit()

        assert Product.query.get(pid) is None
        print("test_delete_product: PASSED")


if __name__ == "__main__":
    test_create_product()
    test_to_dict()
    test_delete_product()
    print("\nAll backend tests passed!")

from flask import Blueprint, jsonify, request

from app.api.schemas import ProductSchema
from app.services.product_service import ProductService

bp = Blueprint("api", __name__, url_prefix="/api")


@bp.route("/products", methods=["GET"])
def get_products():
    products = ProductService.get_all()
    return jsonify(products)


@bp.route("/products/<int:product_id>", methods=["GET"])
def get_product(product_id):
    product = ProductService.get_by_id(product_id)
    if product is None:
        return jsonify({"error": "Product not found"}), 404
    return jsonify(product)


@bp.route("/products", methods=["POST"])
def create_product():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    is_valid, errors = ProductSchema.validate_create(data)
    if not is_valid:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    try:
        product = ProductService.create(data)
        return jsonify(product), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/products/<int:product_id>", methods=["PUT"])
def update_product(product_id):
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    is_valid, errors = ProductSchema.validate_update(data)
    if not is_valid:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    try:
        product = ProductService.update(product_id, data)
        if product is None:
            return jsonify({"error": "Product not found"}), 404
        return jsonify(product)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/products/<int:product_id>", methods=["DELETE"])
def delete_product(product_id):
    deleted = ProductService.delete(product_id)
    if not deleted:
        return jsonify({"error": "Product not found"}), 404
    return jsonify({"message": "Product deleted successfully"}), 200


@bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

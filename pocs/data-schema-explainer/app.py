import os

from flask import Flask, jsonify, request, send_from_directory

from explain import explain

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
SAMPLES = os.path.join(BASE, "sample")

SAMPLE_LABELS = {
    "orders.parquet": "Apache Parquet",
    "orders.arrow": "Apache Arrow / Feather",
    "orders.orc": "Apache ORC",
    "users.avro": "Apache Avro",
    "order.proto": "Protocol Buffers",
    "iceberg_table.metadata.json": "Apache Iceberg",
    "delta_table.log.json": "Delta Lake",
}

app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:path>")
def static_files(path):
    return send_from_directory(STATIC, path)


@app.get("/api/samples")
def samples():
    items = []
    for name, label in SAMPLE_LABELS.items():
        path = os.path.join(SAMPLES, name)
        if os.path.exists(path):
            items.append({"name": name, "label": label, "size": os.path.getsize(path)})
    return jsonify(items)


@app.post("/api/explain")
def explain_endpoint():
    if "file" in request.files:
        f = request.files["file"]
        filename = f.filename
        data = f.read()
    elif request.args.get("sample"):
        name = request.args["sample"]
        if name not in SAMPLE_LABELS:
            return jsonify({"error": "Unknown sample"}), 404
        path = os.path.join(SAMPLES, name)
        filename = name
        with open(path, "rb") as fh:
            data = fh.read()
    else:
        return jsonify({"error": "No file provided"}), 400
    try:
        result = explain(filename, data)
        result["filename"] = filename
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc), "filename": filename}), 422


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))

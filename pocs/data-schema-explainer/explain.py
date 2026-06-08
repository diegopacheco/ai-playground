import io
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather
import pyarrow.orc as orc
import fastavro


def explain_type(type_str):
    t = str(type_str).lower()
    table = [
        ("list", "Ordered collection of values"),
        ("array", "Ordered collection of values"),
        ("repeated", "Ordered collection of values"),
        ("map", "Key to value mapping"),
        ("struct", "Nested record with named fields"),
        ("record", "Nested record with named fields"),
        ("object", "Nested record with named fields"),
        ("decimal", "Fixed-precision decimal number"),
        ("tinyint", "8-bit signed integer"),
        ("smallint", "16-bit signed integer"),
        ("bigint", "64-bit signed integer"),
        ("int8", "8-bit signed integer"),
        ("int16", "16-bit signed integer"),
        ("int32", "32-bit signed integer"),
        ("int64", "64-bit signed integer"),
        ("uint", "Unsigned integer"),
        ("long", "64-bit signed integer"),
        ("integer", "32-bit signed integer"),
        ("double", "64-bit floating point number"),
        ("float", "32-bit floating point number"),
        ("real", "Floating point number"),
        ("boolean", "True or false flag"),
        ("bool", "True or false flag"),
        ("timestamp", "Point in time (date and time)"),
        ("datetime", "Point in time (date and time)"),
        ("date", "Calendar date"),
        ("time", "Time of day"),
        ("uuid", "Universally unique identifier"),
        ("varchar", "Variable length text"),
        ("char", "Text"),
        ("string", "Variable length UTF-8 text"),
        ("utf8", "Variable length UTF-8 text"),
        ("binary", "Raw bytes"),
        ("bytes", "Raw bytes"),
        ("fixed", "Fixed length bytes"),
        ("enum", "One value from a fixed set"),
        ("null", "Always null"),
    ]
    for key, meaning in table:
        if key in t:
            return meaning
    return "Custom or composite type"


def arrow_fields(schema):
    out = []
    for f in schema:
        out.append({
            "name": f.name,
            "type": str(f.type),
            "logicalType": explain_type(f.type),
            "nullable": bool(f.nullable),
            "notes": field_metadata(f.metadata),
        })
    return out


def field_metadata(meta):
    if not meta:
        return ""
    parts = []
    for k, v in meta.items():
        parts.append("{}={}".format(k.decode(), v.decode()))
    return ", ".join(parts)


def schema_metadata(meta):
    if not meta:
        return "none"
    parts = []
    for k, v in meta.items():
        parts.append("{}={}".format(k.decode(), v.decode()))
    return ", ".join(parts)


def explain_parquet(data):
    pf = pq.ParquetFile(io.BytesIO(data))
    md = pf.metadata
    schema = pf.schema_arrow
    compressions = set()
    encodings = set()
    if md.num_row_groups:
        rg = md.row_group(0)
        for i in range(rg.num_columns):
            col = rg.column(i)
            compressions.add(col.compression)
            encodings.update(col.encodings)
    info = [
        ("Rows", "{:,}".format(md.num_rows)),
        ("Columns", md.num_columns),
        ("Row groups", md.num_row_groups),
        ("Compression", ", ".join(sorted(compressions)) or "unknown"),
        ("Column encodings", ", ".join(sorted(encodings)) or "unknown"),
        ("Created by", md.created_by or "unknown"),
        ("Format version", md.format_version),
        ("File size", "{:,} bytes".format(len(data))),
    ]
    return {
        "format": "Apache Parquet",
        "format_key": "parquet",
        "tagline": "Columnar storage file format",
        "description": "Parquet stores data column by column, which makes analytics scans and compression highly efficient. The schema and rich statistics live inside the file footer.",
        "fileInfo": pairs(info),
        "fields": arrow_fields(schema),
        "rawSchema": str(schema),
        "notes": [
            "Data is grouped into row groups, each split into column chunks.",
            "Every column chunk can use its own compression and encoding.",
            "Min/max statistics per column enable predicate push-down at read time.",
        ],
    }


def explain_arrow(data):
    reader = pa.ipc.open_file(pa.BufferReader(data))
    schema = reader.schema
    table = reader.read_all()
    info = [
        ("Rows", "{:,}".format(table.num_rows)),
        ("Columns", len(schema.names)),
        ("Record batches", reader.num_record_batches),
        ("Schema metadata", schema_metadata(schema.metadata)),
        ("File size", "{:,} bytes".format(len(data))),
    ]
    return {
        "format": "Apache Arrow / Feather",
        "format_key": "arrow",
        "tagline": "In-memory columnar format on disk",
        "description": "Arrow defines a language independent columnar memory layout. The Feather (Arrow IPC) file is that same layout written to disk, so it loads with zero deserialization cost.",
        "fileInfo": pairs(info),
        "fields": arrow_fields(schema),
        "rawSchema": str(schema),
        "notes": [
            "Columns are stored as record batches sharing one schema.",
            "The on-disk layout matches the in-memory layout, so reads need no parsing.",
            "Designed for fast zero-copy data exchange between tools and languages.",
        ],
    }


def explain_orc(data):
    f = orc.ORCFile(io.BytesIO(data))
    schema = f.schema
    info = [
        ("Rows", "{:,}".format(f.nrows)),
        ("Columns", len(schema.names)),
        ("Stripes", f.nstripes),
        ("Compression", str(getattr(f, "compression", "unknown"))),
        ("Compression block size", "{:,}".format(getattr(f, "compression_size", 0))),
        ("Row index stride", "{:,}".format(getattr(f, "row_index_stride", 0))),
        ("File version", str(getattr(f, "file_version", "unknown"))),
        ("File size", "{:,} bytes".format(len(data))),
    ]
    return {
        "format": "Apache ORC",
        "format_key": "orc",
        "tagline": "Optimized Row Columnar format",
        "description": "ORC is a columnar format built for the Hadoop and Hive world. It splits data into stripes, keeps lightweight indexes and per-column statistics, and compresses aggressively.",
        "fileInfo": pairs(info),
        "fields": arrow_fields(schema),
        "rawSchema": str(schema),
        "notes": [
            "Rows are grouped into stripes, each holding index, data and footer sections.",
            "Built-in indexes let readers skip stripes and row groups that fail a filter.",
            "Originated in Apache Hive as a successor to the RCFile format.",
        ],
    }


def avro_type_label(t):
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        return " | ".join(avro_type_label(x) for x in non_null) or "null"
    if isinstance(t, dict):
        if t.get("logicalType"):
            return "{} ({})".format(t.get("type"), t["logicalType"])
        kind = t.get("type")
        if kind == "array":
            return "array<{}>".format(avro_type_label(t.get("items")))
        if kind == "map":
            return "map<string,{}>".format(avro_type_label(t.get("values")))
        if kind == "record":
            return "record {}".format(t.get("name", ""))
        if kind == "enum":
            return "enum {}".format("|".join(t.get("symbols", [])))
        return str(kind)
    return str(t)


def avro_nullable(t):
    return isinstance(t, list) and "null" in t


def explain_avro(data):
    reader = fastavro.reader(io.BytesIO(data))
    schema = reader.writer_schema
    fields = []
    for f in schema.get("fields", []):
        t = f.get("type")
        fields.append({
            "name": f.get("name"),
            "type": avro_type_label(t),
            "logicalType": explain_type(avro_type_label(t)),
            "nullable": avro_nullable(t),
            "notes": "default={}".format(json.dumps(f["default"])) if "default" in f else "",
        })
    info = [
        ("Record name", schema.get("name", "")),
        ("Namespace", schema.get("namespace", "")),
        ("Codec", reader.metadata.get("avro.codec", "null")),
        ("Top level fields", len(schema.get("fields", []))),
        ("File size", "{:,} bytes".format(len(data))),
    ]
    return {
        "format": "Apache Avro",
        "format_key": "avro",
        "tagline": "Row based format with embedded schema",
        "description": "Avro stores records row by row and writes the full schema into the file header as JSON. That self-describing header makes Avro a popular choice for streaming and message payloads.",
        "fileInfo": pairs(info),
        "fields": fields,
        "rawSchema": json.dumps(schema, indent=2),
        "notes": [
            "The writer schema travels with the data, so readers never guess the layout.",
            "Nullable fields are modelled as a union with the null type.",
            "Schema evolution rules let new readers consume data from old writers.",
        ],
    }


def explain_protobuf(text):
    syntax = re.search(r'syntax\s*=\s*"([^"]+)"', text)
    package = re.search(r'package\s+([\w\.]+)\s*;', text)
    messages = re.findall(r'message\s+(\w+)\s*\{([^}]*)\}', text)
    enums = re.findall(r'enum\s+(\w+)\s*\{([^}]*)\}', text)
    field_re = re.compile(r'(repeated|optional|required)?\s*([\w\.]+)\s+(\w+)\s*=\s*(\d+)\s*;')
    fields = []
    for msg_name, body in messages:
        for label, ftype, fname, number in field_re.findall(body):
            fields.append({
                "name": "{}.{}".format(msg_name, fname),
                "type": (label + " " if label else "") + ftype,
                "logicalType": explain_type(ftype),
                "nullable": label in ("optional",),
                "notes": "field #{} in {}".format(number, msg_name),
            })
    info = [
        ("Syntax", syntax.group(1) if syntax else "proto2"),
        ("Package", package.group(1) if package else ""),
        ("Messages", ", ".join(m[0] for m in messages) or "none"),
        ("Enums", ", ".join(e[0] for e in enums) or "none"),
        ("File size", "{:,} bytes".format(len(text.encode()))),
    ]
    return {
        "format": "Protocol Buffers",
        "format_key": "protobuf",
        "tagline": "Schema definition language for binary messages",
        "description": "A .proto file declares messages and their fields. Each field carries a stable tag number used in the compact binary wire format, which is what gets sent over the network.",
        "fileInfo": pairs(info),
        "fields": fields,
        "rawSchema": text.strip(),
        "notes": [
            "Field numbers are the identity on the wire and must never be reused.",
            "Encoded messages do not contain field names, only tag numbers.",
            "repeated means a list, optional marks presence tracking in proto3.",
        ],
    }


def iceberg_type_label(t):
    if isinstance(t, str):
        return t
    if isinstance(t, dict):
        kind = t.get("type")
        if kind == "struct":
            return "struct<{} fields>".format(len(t.get("fields", [])))
        if kind == "list":
            return "list<{}>".format(iceberg_type_label(t.get("element")))
        if kind == "map":
            return "map<{},{}>".format(iceberg_type_label(t.get("key")), iceberg_type_label(t.get("value")))
        return str(kind)
    return str(t)


def explain_iceberg(obj):
    schemas = obj.get("schemas")
    if not schemas:
        schemas = [obj.get("schema")] if obj.get("schema") else []
    current_id = obj.get("current-schema-id", 0)
    schema = next((s for s in schemas if s.get("schema-id") == current_id), schemas[0] if schemas else {})
    fields = []
    for f in schema.get("fields", []):
        fields.append({
            "name": f.get("name"),
            "type": iceberg_type_label(f.get("type")),
            "logicalType": explain_type(iceberg_type_label(f.get("type"))),
            "nullable": not f.get("required", False),
            "notes": "field id {}".format(f.get("id")),
        })
    specs = obj.get("partition-specs", [])
    part_fields = []
    for spec in specs:
        for pf in spec.get("fields", []):
            part_fields.append("{}({})".format(pf.get("transform"), pf.get("name")))
    info = [
        ("Format version", obj.get("format-version", "")),
        ("Table UUID", obj.get("table-uuid", "")),
        ("Location", obj.get("location", "")),
        ("Current schema id", current_id),
        ("Partitioning", ", ".join(part_fields) or "unpartitioned"),
        ("Snapshots", len(obj.get("snapshots", []))),
        ("Properties", ", ".join("{}={}".format(k, v) for k, v in obj.get("properties", {}).items()) or "none"),
    ]
    return {
        "format": "Apache Iceberg",
        "format_key": "iceberg",
        "tagline": "Open table format metadata",
        "description": "Iceberg is a table format, not a file format. This metadata JSON tracks the table schema, partitioning and a history of immutable snapshots that point to the underlying data files.",
        "fileInfo": pairs(info),
        "fields": fields,
        "rawSchema": json.dumps(schema, indent=2),
        "notes": [
            "Each field has a stable integer id, so columns can be renamed safely.",
            "Snapshots make atomic commits and time travel queries possible.",
            "Partitioning is hidden: queries do not need partition columns spelled out.",
        ],
    }


def explain_delta(text):
    actions = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            actions.append(json.loads(line))
    meta = next((a["metaData"] for a in actions if "metaData" in a), {})
    protocol = next((a["protocol"] for a in actions if "protocol" in a), {})
    adds = [a for a in actions if "add" in a]
    spark_schema = json.loads(meta.get("schemaString", "{}"))
    fields = []
    for f in spark_schema.get("fields", []):
        fields.append({
            "name": f.get("name"),
            "type": delta_type_label(f.get("type")),
            "logicalType": explain_type(delta_type_label(f.get("type"))),
            "nullable": bool(f.get("nullable", True)),
            "notes": field_meta_label(f.get("metadata")),
        })
    info = [
        ("Table id", meta.get("id", "")),
        ("Provider", meta.get("format", {}).get("provider", "")),
        ("Partition columns", ", ".join(meta.get("partitionColumns", [])) or "none"),
        ("Data files in commit", len(adds)),
        ("Min reader version", protocol.get("minReaderVersion", "")),
        ("Min writer version", protocol.get("minWriterVersion", "")),
        ("Configuration", ", ".join("{}={}".format(k, v) for k, v in meta.get("configuration", {}).items()) or "none"),
    ]
    return {
        "format": "Delta Lake",
        "format_key": "delta",
        "tagline": "Transaction log over Parquet",
        "description": "Delta Lake is a table format layered on Parquet files. This JSON line comes from the _delta_log: an ordered transaction log whose actions define the schema and which data files are live.",
        "fileInfo": pairs(info),
        "fields": fields,
        "rawSchema": json.dumps(spark_schema, indent=2),
        "notes": [
            "Each commit is one JSON file of actions in the _delta_log folder.",
            "The metaData action holds the schema as a serialized Spark struct.",
            "add and remove actions track which Parquet files belong to the table.",
        ],
    }


def delta_type_label(t):
    if isinstance(t, str):
        return t
    if isinstance(t, dict):
        kind = t.get("type")
        if kind == "array":
            return "array<{}>".format(delta_type_label(t.get("elementType")))
        if kind == "map":
            return "map<{},{}>".format(delta_type_label(t.get("keyType")), delta_type_label(t.get("valueType")))
        if kind == "struct":
            return "struct<{} fields>".format(len(t.get("fields", [])))
        return str(kind)
    return str(t)


def field_meta_label(meta):
    if not meta:
        return ""
    return ", ".join("{}={}".format(k, v) for k, v in meta.items())


def pairs(items):
    return [{"label": k, "value": str(v)} for k, v in items]


def detect_format(filename, data):
    name = (filename or "").lower()
    if data[:4] == b"PAR1" or data[-4:] == b"PAR1" or name.endswith(".parquet"):
        return "parquet"
    if data[:6] == b"ARROW1" or name.endswith((".arrow", ".feather", ".ipc")):
        return "arrow"
    if data[:4] == b"Obj\x01" or name.endswith(".avro"):
        return "avro"
    if data[:3] == b"ORC" or name.endswith(".orc"):
        return "orc"
    if name.endswith(".proto"):
        return "protobuf"
    text = None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    stripped = text.lstrip()
    if stripped.startswith("syntax") or "message " in text and "= " in text and not stripped.startswith("{"):
        if "message" in text and "{" in text:
            return "protobuf"
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and ("format-version" in obj or "table-uuid" in obj):
            return "iceberg"
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(key in line for key in ('"metaData"', '"protocol"', '"commitInfo"', '"add"')):
            return "delta"
        break
    return None


def explain(filename, data):
    fmt = detect_format(filename, data)
    if fmt == "parquet":
        return explain_parquet(data)
    if fmt == "arrow":
        return explain_arrow(data)
    if fmt == "orc":
        return explain_orc(data)
    if fmt == "avro":
        return explain_avro(data)
    if fmt == "protobuf":
        return explain_protobuf(data.decode("utf-8"))
    if fmt == "iceberg":
        return explain_iceberg(json.loads(data.decode("utf-8")))
    if fmt == "delta":
        return explain_delta(data.decode("utf-8"))
    raise ValueError("Unsupported or unrecognized file format")

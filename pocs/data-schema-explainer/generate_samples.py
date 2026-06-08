import datetime
import decimal
import json
import os
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather
import pyarrow.orc as orc
import fastavro

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "sample")


def orders_table():
    schema = pa.schema([
        pa.field("order_id", pa.int64(), nullable=False),
        pa.field("customer", pa.string()),
        pa.field("total", pa.decimal128(10, 2)),
        pa.field("currency", pa.string()),
        pa.field("paid", pa.bool_()),
        pa.field("created_at", pa.timestamp("us")),
        pa.field("items", pa.list_(pa.string())),
        pa.field("shipping", pa.struct([
            pa.field("city", pa.string()),
            pa.field("zip", pa.string()),
        ])),
    ])
    rows = {
        "order_id": [1001, 1002, 1003, 1004],
        "customer": ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Edsger Dijkstra"],
        "total": [decimal.Decimal("129.90"), decimal.Decimal("59.00"),
                  decimal.Decimal("1499.99"), decimal.Decimal("12.50")],
        "currency": ["USD", "USD", "EUR", "BRL"],
        "paid": [True, False, True, True],
        "created_at": [
            datetime.datetime(2026, 1, 5, 9, 30),
            datetime.datetime(2026, 2, 14, 18, 5),
            datetime.datetime(2026, 3, 1, 11, 45),
            datetime.datetime(2026, 3, 22, 8, 0),
        ],
        "items": [["keyboard", "mouse"], ["book"], ["gpu", "cable", "fan"], ["sticker"]],
        "shipping": [
            {"city": "London", "zip": "EC1A"},
            {"city": "Manchester", "zip": "M1"},
            {"city": "Berlin", "zip": "10115"},
            {"city": "Sao Paulo", "zip": "01310"},
        ],
    }
    return pa.Table.from_pydict(rows, schema=schema)


def write_parquet(table):
    pq.write_table(table, os.path.join(OUT, "orders.parquet"), compression="snappy")


def write_arrow(table):
    feather.write_feather(table, os.path.join(OUT, "orders.arrow"))


def write_orc(table):
    orc.write_table(table, os.path.join(OUT, "orders.orc"))


def write_avro():
    schema = {
        "type": "record",
        "name": "User",
        "namespace": "com.shop",
        "fields": [
            {"name": "user_id", "type": "long"},
            {"name": "email", "type": "string"},
            {"name": "name", "type": ["null", "string"], "default": None},
            {"name": "age", "type": ["null", "int"], "default": None},
            {"name": "active", "type": "boolean"},
            {"name": "signup_ts", "type": {"type": "long", "logicalType": "timestamp-millis"}},
            {"name": "roles", "type": {"type": "array", "items": "string"}},
            {"name": "address", "type": {
                "type": "record",
                "name": "Address",
                "fields": [
                    {"name": "city", "type": "string"},
                    {"name": "zip", "type": "string"},
                ],
            }},
        ],
    }
    records = [
        {"user_id": 1, "email": "ada@shop.io", "name": "Ada", "age": 36, "active": True,
         "signup_ts": 1767225600000, "roles": ["admin", "buyer"],
         "address": {"city": "London", "zip": "EC1A"}},
        {"user_id": 2, "email": "alan@shop.io", "name": None, "age": None, "active": False,
         "signup_ts": 1771200000000, "roles": ["buyer"],
         "address": {"city": "Manchester", "zip": "M1"}},
    ]
    with open(os.path.join(OUT, "users.avro"), "wb") as fh:
        fastavro.writer(fh, schema, records, codec="deflate")


def write_proto():
    text = '''syntax = "proto3";

package shop.orders;

enum Currency {
  CURRENCY_UNSPECIFIED = 0;
  USD = 1;
  EUR = 2;
  BRL = 3;
}

message Address {
  string city = 1;
  string zip = 2;
}

message Order {
  int64 order_id = 1;
  string customer = 2;
  double total = 3;
  Currency currency = 4;
  bool paid = 5;
  int64 created_at = 6;
  repeated string items = 7;
  Address shipping = 8;
  optional string coupon = 9;
}
'''
    with open(os.path.join(OUT, "order.proto"), "w") as fh:
        fh.write(text)


def write_iceberg():
    schema = {
        "type": "struct",
        "schema-id": 0,
        "fields": [
            {"id": 1, "name": "order_id", "required": True, "type": "long"},
            {"id": 2, "name": "customer", "required": False, "type": "string"},
            {"id": 3, "name": "total", "required": False, "type": "decimal(10, 2)"},
            {"id": 4, "name": "currency", "required": False, "type": "string"},
            {"id": 5, "name": "paid", "required": False, "type": "boolean"},
            {"id": 6, "name": "created_at", "required": False, "type": "timestamptz"},
            {"id": 7, "name": "items", "required": False, "type": {
                "type": "list", "element-id": 10, "element": "string", "element-required": False}},
            {"id": 8, "name": "shipping", "required": False, "type": {
                "type": "struct", "fields": [
                    {"id": 11, "name": "city", "required": False, "type": "string"},
                    {"id": 12, "name": "zip", "required": False, "type": "string"},
                ]}},
        ],
    }
    metadata = {
        "format-version": 2,
        "table-uuid": str(uuid.uuid4()),
        "location": "s3://warehouse/shop/orders",
        "last-updated-ms": 1772000000000,
        "last-column-id": 12,
        "current-schema-id": 0,
        "schemas": [schema],
        "default-spec-id": 0,
        "partition-specs": [{
            "spec-id": 0,
            "fields": [{"name": "created_at_day", "transform": "day",
                        "source-id": 6, "field-id": 1000}],
        }],
        "default-sort-order-id": 0,
        "sort-orders": [{"order-id": 0, "fields": []}],
        "properties": {"write.format.default": "parquet", "owner": "data-platform"},
        "current-snapshot-id": 3055729675574597004,
        "snapshots": [{
            "snapshot-id": 3055729675574597004,
            "timestamp-ms": 1772000000000,
            "summary": {"operation": "append", "added-data-files": "4",
                        "added-records": "4"},
            "manifest-list": "s3://warehouse/shop/orders/metadata/snap-3055.avro",
            "schema-id": 0,
        }],
        "snapshot-log": [{"snapshot-id": 3055729675574597004,
                          "timestamp-ms": 1772000000000}],
        "metadata-log": [],
    }
    with open(os.path.join(OUT, "iceberg_table.metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)


def write_delta():
    spark_schema = {
        "type": "struct",
        "fields": [
            {"name": "order_id", "type": "long", "nullable": False, "metadata": {}},
            {"name": "customer", "type": "string", "nullable": True, "metadata": {}},
            {"name": "total", "type": "decimal(10,2)", "nullable": True, "metadata": {}},
            {"name": "currency", "type": "string", "nullable": True, "metadata": {}},
            {"name": "paid", "type": "boolean", "nullable": True, "metadata": {}},
            {"name": "created_at", "type": "timestamp", "nullable": True, "metadata": {}},
            {"name": "items", "type": {"type": "array", "elementType": "string",
                                       "containsNull": True}, "nullable": True, "metadata": {}},
            {"name": "shipping", "type": {"type": "struct", "fields": [
                {"name": "city", "type": "string", "nullable": True, "metadata": {}},
                {"name": "zip", "type": "string", "nullable": True, "metadata": {}},
            ]}, "nullable": True, "metadata": {}},
        ],
    }
    actions = [
        {"commitInfo": {"timestamp": 1772000000000, "operation": "WRITE",
                        "operationParameters": {"mode": "Append"},
                        "engineInfo": "Apache-Spark/3.5"}},
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        {"metaData": {
            "id": str(uuid.uuid4()),
            "format": {"provider": "parquet", "options": {}},
            "schemaString": json.dumps(spark_schema),
            "partitionColumns": ["currency"],
            "configuration": {"delta.appendOnly": "true"},
            "createdTime": 1772000000000,
        }},
        {"add": {"path": "currency=USD/part-00000.snappy.parquet", "partitionValues": {"currency": "USD"},
                 "size": 1234, "modificationTime": 1772000000000, "dataChange": True}},
        {"add": {"path": "currency=EUR/part-00001.snappy.parquet", "partitionValues": {"currency": "EUR"},
                 "size": 980, "modificationTime": 1772000000000, "dataChange": True}},
        {"add": {"path": "currency=BRL/part-00002.snappy.parquet", "partitionValues": {"currency": "BRL"},
                 "size": 870, "modificationTime": 1772000000000, "dataChange": True}},
    ]
    with open(os.path.join(OUT, "delta_table.log.json"), "w") as fh:
        for action in actions:
            fh.write(json.dumps(action) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    table = orders_table()
    write_parquet(table)
    write_arrow(table)
    write_orc(table)
    write_avro()
    write_proto()
    write_iceberg()
    write_delta()
    for name in sorted(os.listdir(OUT)):
        path = os.path.join(OUT, name)
        if os.path.isfile(path):
            print("{:32} {:>8} bytes".format(name, os.path.getsize(path)))


if __name__ == "__main__":
    main()

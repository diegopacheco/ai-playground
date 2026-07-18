import { allEngines, engineFor } from "./index";
import type { SchemaNode } from "@lib/types";

describe("engine descriptors", () => {
  it("covers every connection kind the backend can return", () => {
    const kinds = allEngines.map((engine) => engine.kind).sort();
    expect(kinds).toEqual([
      "cassandra", "elasticsearch", "etcd", "kafka", "mysql", "postgres", "redis"
    ]);
  });

  it("offers a row count only where counting is cheap, so a console never triggers a full cluster scan", () => {
    expect(engineFor("postgres").supportsCount).toBe(true);
    expect(engineFor("mysql").supportsCount).toBe(true);
    expect(engineFor("cassandra").supportsCount).toBe(false);
    expect(engineFor("kafka").supportsCount).toBe(false);
    expect(engineFor("elasticsearch").supportsCount).toBe(false);
    expect(engineFor("redis").supportsCount).toBe(false);
    expect(engineFor("etcd").supportsCount).toBe(false);
  });

  it("builds a sample statement in each engine's own grammar", () => {
    const table: SchemaNode[] = [{ name: "customers", kind: "table" }];
    expect(engineFor("postgres").sampleStatement(table)).toBe("SELECT * FROM customers");
    expect(engineFor("kafka").sampleStatement([{ name: "orders.events", kind: "topic" }]))
      .toBe("consume orders.events --limit 100");
    expect(engineFor("etcd").sampleStatement([{ name: "config", kind: "prefix" }]))
      .toBe("get /config --prefix");
    expect(engineFor("elasticsearch").sampleStatement([{ name: "products", kind: "index" }]))
      .toBe("GET /products/_search");
    expect(engineFor("redis").sampleStatement([{ name: "session:a", kind: "hash" }]))
      .toBe("HGETALL session:a");
  });

  it("picks a redis command that matches the key type, since GET on a hash is an error", () => {
    expect(engineFor("redis").sampleStatement([{ name: "counter", kind: "string" }])).toBe("GET counter");
    expect(engineFor("redis").sampleStatement([{ name: "h", kind: "hash" }])).toBe("HGETALL h");
  });

  it("stays usable when a connection has no objects at all", () => {
    allEngines.forEach((engine) => {
      expect(engine.sampleStatement([]).length).toBeGreaterThan(0);
      expect(engine.emptySchemaLabel.length).toBeGreaterThan(0);
    });
  });

  it("feeds real schema names into completions so autocomplete is never invented", () => {
    const schema: SchemaNode[] = [
      { name: "customers", kind: "table", children: [{ name: "email", kind: "column" }] }
    ];
    const completions = engineFor("postgres").completionsFor(schema);
    expect(completions).toContain("customers");
    expect(completions).toContain("email");
    expect(completions).toContain("SELECT");
  });

  it("builds full etcd paths for completion, because a bare segment is not addressable", () => {
    const schema: SchemaNode[] = [
      { name: "config", kind: "prefix", children: [{ name: "app", kind: "prefix", children: [{ name: "name", kind: "key" }] }] }
    ];
    expect(engineFor("etcd").completionsFor(schema)).toContain("/config/app/name");
  });
});

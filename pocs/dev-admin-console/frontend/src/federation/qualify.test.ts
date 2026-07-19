import { aliasFor, insertAt, qualify } from "./qualify";

const statement = `SELECT a.id
FROM demo-mysql.invoices a
JOIN demo-elasticsearch.products b ON a.id = b._id
LIMIT 25`;

describe("qualify", () => {
  it("inserts a connection name on its own", () => {
    expect(qualify(statement, ["demo-mysql"])).toBe("demo-mysql");
  });

  it("qualifies a source with its connection, which is what FROM and JOIN need", () => {
    expect(qualify(statement, ["demo-mysql", "invoices"])).toBe("demo-mysql.invoices");
  });

  it("qualifies a column with the alias already used for that source", () => {
    expect(qualify(statement, ["demo-mysql", "invoices", "number"])).toBe("a.number");
    expect(qualify(statement, ["demo-elasticsearch", "products", "name"])).toBe("b.name");
  });

  it("falls back to the bare column when that source has no alias yet", () => {
    expect(qualify(statement, ["demo-postgres", "orders", "status"])).toBe("status");
  });

  it("handles sources containing dots, slashes and colons", () => {
    const kafka = "SELECT x.key FROM demo-kafka.orders.events x JOIN demo-etcd./config y ON x.key = y.key";
    expect(aliasFor(kafka, "demo-kafka", "orders.events")).toBe("x");
    expect(aliasFor(kafka, "demo-etcd", "/config")).toBe("y");
    expect(qualify(kafka, ["demo-etcd", "/config", "value"])).toBe("y.value");
  });

  it("does not mistake a keyword for an alias", () => {
    const noAlias = "SELECT * FROM demo-mysql.invoices JOIN demo-etcd.config ON a.id = b.id";
    expect(aliasFor(noAlias, "demo-mysql", "invoices")).toBeNull();
  });

  it("reads an explicit AS alias", () => {
    expect(aliasFor("FROM demo-mysql.invoices AS inv", "demo-mysql", "invoices")).toBe("inv");
  });
});

describe("insertAt", () => {
  it("adds a space when the statement does not end in whitespace", () => {
    expect(insertAt("SELECT", "a.id")).toBe("SELECT a.id");
  });

  it("does not double the space", () => {
    expect(insertAt("SELECT ", "a.id")).toBe("SELECT a.id");
    expect(insertAt("SELECT\n", "a.id")).toBe("SELECT\na.id");
  });

  it("starts an empty statement cleanly", () => {
    expect(insertAt("", "demo-mysql.invoices")).toBe("demo-mysql.invoices");
  });
});

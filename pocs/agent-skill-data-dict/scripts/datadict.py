#!/usr/bin/env python3
import os
import re
import sys
import json
import html
import datetime
import xml.etree.ElementTree as ET
from collections import OrderedDict

SKIP_DIRS = {
    ".git", "target", "build", "out", "bin", "obj", "node_modules", "dist",
    ".idea", ".gradle", ".mvn", ".venv", "venv", "__pycache__", ".next",
    "coverage", ".terraform", "vendor",
}

COL_OPTION_KEYWORDS = {
    "NOT", "NULL", "PRIMARY", "KEY", "DEFAULT", "UNIQUE", "REFERENCES",
    "AUTO_INCREMENT", "AUTOINCREMENT", "IDENTITY", "GENERATED", "CHECK",
    "COLLATE", "COMMENT", "CONSTRAINT", "FOREIGN",
}

TYPE_WIDTH = {
    "BIGINT": 8, "INT": 4, "INTEGER": 4, "SMALLINT": 2, "TINYINT": 1,
    "BOOLEAN": 1, "BOOL": 1, "DOUBLE": 8, "FLOAT": 8, "REAL": 4,
    "DATE": 4, "TIME": 4, "TIMESTAMP": 8, "DATETIME": 8,
    "TEXT": 16, "CLOB": 16, "BLOB": 16, "UUID": 16,
}

JAVA_TYPE_MAP = {
    "long": "BIGINT", "Long": "BIGINT",
    "int": "INT", "Integer": "INT", "short": "SMALLINT", "Short": "SMALLINT",
    "BigInteger": "NUMERIC(38)",
    "BigDecimal": "NUMERIC", "double": "DOUBLE", "Double": "DOUBLE",
    "float": "FLOAT", "Float": "FLOAT",
    "boolean": "BOOLEAN", "Boolean": "BOOLEAN",
    "String": "VARCHAR", "char": "CHAR", "Character": "CHAR",
    "UUID": "VARCHAR(36)",
    "LocalDate": "DATE", "LocalDateTime": "TIMESTAMP", "Instant": "TIMESTAMP",
    "OffsetDateTime": "TIMESTAMP", "ZonedDateTime": "TIMESTAMP",
    "Date": "TIMESTAMP", "Timestamp": "TIMESTAMP", "LocalTime": "TIME",
}


def snake(name):
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def new_table(name):
    return {
        "name": name,
        "cols": OrderedDict(),
        "pk": [],
        "fks": [],
        "sources": [],
        "origins": [],
        "remarks": "",
    }


def new_col(name):
    return {
        "name": name, "type": "", "base": "", "size": None,
        "precision": None, "scale": None, "nullable": True, "pk": False,
        "unique": False, "default": None, "remarks": "", "fk": None,
        "source": "",
    }


def get_table(model, name):
    key = name.strip().strip('"').strip("`").split(".")[-1].lower()
    if key not in model:
        model[key] = new_table(name.strip().strip('"').strip("`").split(".")[-1])
    return model[key]


def add_source(table, source, origin):
    if source not in table["sources"]:
        table["sources"].append(source)
    if origin and origin not in table["origins"]:
        table["origins"].append(origin)


def upsert_col(table, col, source):
    key = col["name"].lower()
    existing = table["cols"].get(key)
    if existing is None:
        col["source"] = source
        table["cols"][key] = col
        if col["pk"] and col["name"] not in table["pk"]:
            table["pk"].append(col["name"])
        return col
    for f in ("type", "base", "size", "precision", "scale", "default", "remarks", "fk"):
        if not existing.get(f) and col.get(f):
            existing[f] = col[f]
    if col["pk"]:
        existing["pk"] = True
        if existing["name"] not in table["pk"]:
            table["pk"].append(existing["name"])
    if not col["nullable"]:
        existing["nullable"] = False
    if col["unique"]:
        existing["unique"] = True
    return existing


def parse_type(typestr):
    if not typestr:
        return ("", None, None, None)
    t = typestr.strip()
    m = re.match(r"^([A-Za-z_][\w ]*?)\s*\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)", t)
    if m:
        base = m.group(1).strip().upper()
        size = int(m.group(2))
        scale = int(m.group(3)) if m.group(3) else None
        if scale is not None:
            return (base, None, size, scale)
        return (base, size, None, None)
    base = re.split(r"\s+", t)[0].upper()
    return (base, None, None, None)


def set_col_type(col, typestr):
    base, size, precision, scale = parse_type(typestr)
    col["base"] = base
    col["size"] = size
    col["precision"] = precision
    col["scale"] = scale
    if precision is not None and scale is not None:
        col["type"] = "%s(%d,%d)" % (base, precision, scale)
    elif precision is not None:
        col["type"] = "%s(%d)" % (base, precision)
    elif size is not None:
        col["type"] = "%s(%d)" % (base, size)
    else:
        col["type"] = base or (typestr or "").upper()


def parse_references(ref):
    m = re.match(r"\s*([\w\"`]+)\s*(?:\(\s*([\w\"`]+)\s*\))?", ref or "")
    if not m:
        return None
    table = m.group(1).strip('"').strip("`")
    column = (m.group(2) or "id").strip('"').strip("`")
    return {"table": table, "column": column}


def split_top_level(body):
    parts = []
    depth = 0
    cur = []
    quote = None
    for ch in body:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"`":
            quote = ch
            cur.append(ch)
            continue
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def tokenize_coldef(text):
    tokens = []
    cur = []
    depth = 0
    quote = None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"`":
            quote = ch
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch.isspace() and depth == 0:
            if cur:
                tokens.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        tokens.append("".join(cur))
    return tokens


def parse_create_table(blob):
    results = []
    upper = blob.upper()
    idx = 0
    while True:
        pos = upper.find("CREATE TABLE", idx)
        if pos == -1:
            break
        rest = blob[pos + len("CREATE TABLE"):]
        rest_stripped = rest.lstrip()
        offset = len(rest) - len(rest_stripped)
        m = re.match(r"(?:IF\s+NOT\s+EXISTS\s+)?([\w\"`.]+)", rest_stripped, re.IGNORECASE)
        if not m:
            idx = pos + len("CREATE TABLE")
            continue
        name = m.group(1)
        after = rest_stripped[m.end():]
        paren = after.find("(")
        if paren == -1:
            idx = pos + len("CREATE TABLE")
            continue
        depth = 0
        body_chars = []
        started = False
        end_index = None
        for i, ch in enumerate(after[paren:]):
            if ch == "(":
                depth += 1
                started = True
                if depth == 1:
                    continue
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end_index = i
                    break
            if started:
                body_chars.append(ch)
        if end_index is None:
            idx = pos + len("CREATE TABLE")
            continue
        body = "".join(body_chars)
        results.append((name, body))
        consumed = pos + len("CREATE TABLE") + offset + m.end() + paren + end_index
        idx = consumed + 1
    return results


def apply_table_body(table, body, source):
    for part in split_top_level(body):
        if not part.strip():
            continue
        upper = part.upper().lstrip()
        cons = re.match(r"(?:CONSTRAINT\s+[\w\"`]+\s+)?(PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CHECK|KEY|INDEX)\b",
                        upper, re.IGNORECASE)
        if cons:
            kind = re.sub(r"\s+", " ", cons.group(1).upper())
            if kind == "PRIMARY KEY":
                cols = re.search(r"\(([^)]*)\)", part)
                if cols:
                    for c in cols.group(1).split(","):
                        cn = c.strip().strip('"').strip("`")
                        if cn:
                            ex = table["cols"].get(cn.lower())
                            if ex:
                                ex["pk"] = True
                            if cn not in table["pk"]:
                                table["pk"].append(cn)
            elif kind == "FOREIGN KEY":
                fk = re.search(r"FOREIGN\s+KEY\s*\(([^)]*)\)\s*REFERENCES\s+([\w\"`.]+)\s*(?:\(([^)]*)\))?",
                               part, re.IGNORECASE)
                if fk:
                    base_cols = [c.strip().strip('"').strip("`") for c in fk.group(1).split(",")]
                    ref_table = fk.group(2).strip('"').strip("`").split(".")[-1]
                    ref_cols = [c.strip().strip('"').strip("`") for c in (fk.group(3) or "id").split(",")]
                    table["fks"].append({
                        "columns": base_cols, "ref_table": ref_table,
                        "ref_columns": ref_cols, "inferred": False,
                    })
                    for bc in base_cols:
                        ex = table["cols"].get(bc.lower())
                        if ex and not ex["fk"]:
                            ex["fk"] = {"table": ref_table, "column": ref_cols[0] if ref_cols else "id"}
            elif kind == "UNIQUE":
                cols = re.search(r"\(([^)]*)\)", part)
                if cols:
                    for c in cols.group(1).split(","):
                        ex = table["cols"].get(c.strip().strip('"').strip("`").lower())
                        if ex:
                            ex["unique"] = True
            continue

        tokens = tokenize_coldef(part)
        if not tokens:
            continue
        col = new_col(tokens[0].strip('"').strip("`"))
        if len(tokens) >= 2:
            set_col_type(col, tokens[1])
        rest = " ".join(tokens[2:]).upper() if len(tokens) > 2 else ""
        original_rest = " ".join(tokens[2:]) if len(tokens) > 2 else ""
        if "NOT NULL" in rest:
            col["nullable"] = False
        if "PRIMARY KEY" in rest:
            col["pk"] = True
            col["nullable"] = False
        if re.search(r"\bUNIQUE\b", rest):
            col["unique"] = True
        if re.search(r"AUTO_INCREMENT|AUTOINCREMENT|IDENTITY|GENERATED", rest):
            col["default"] = "auto-increment"
        dm = re.search(r"DEFAULT\s+('(?:[^']*)'|[\w.+-]+|CURRENT_TIMESTAMP)", original_rest, re.IGNORECASE)
        if dm:
            col["default"] = dm.group(1).strip("'")
        ref = re.search(r"REFERENCES\s+([\w\"`.]+)\s*(?:\(([^)]*)\))?", original_rest, re.IGNORECASE)
        if ref:
            ref_table = ref.group(1).strip('"').strip("`").split(".")[-1]
            ref_col = (ref.group(2) or "id").strip().strip('"').strip("`")
            col["fk"] = {"table": ref_table, "column": ref_col}
            table["fks"].append({
                "columns": [col["name"]], "ref_table": ref_table,
                "ref_columns": [ref_col], "inferred": False,
            })
        upsert_col(table, col, source)


def parse_sql_blob(model, blob, source, origin):
    for name, body in parse_create_table(blob):
        table = get_table(model, name)
        add_source(table, source, origin)
        apply_table_body(table, body, source)
    for m in re.finditer(r"ALTER\s+TABLE\s+([\w\"`.]+)\s+ADD\s+(?:COLUMN\s+)?([^;]+)",
                         blob, re.IGNORECASE):
        tname = m.group(1).strip('"').strip("`").split(".")[-1]
        spec = m.group(2).strip()
        if spec.upper().startswith(("CONSTRAINT", "PRIMARY", "FOREIGN", "UNIQUE")):
            continue
        table = get_table(model, tname)
        add_source(table, source, origin)
        apply_table_body(table, spec, source)


def localname(tag):
    return tag.split("}")[-1]


def parse_liquibase_xml(model, path, rel):
    try:
        tree = ET.parse(path)
    except Exception:
        return False
    root = tree.getroot()
    if localname(root.tag) != "databaseChangeLog":
        return False
    for change_set in root:
        if localname(change_set.tag) != "changeSet":
            continue
        cs_id = change_set.get("id", "")
        origin = "%s (changeSet %s)" % (rel, cs_id)
        for change in change_set:
            tag = localname(change.tag)
            if tag == "createTable":
                table = get_table(model, change.get("tableName", ""))
                add_source(table, "liquibase", origin)
                if change.get("remarks"):
                    table["remarks"] = change.get("remarks")
                for column in change:
                    if localname(column.tag) == "column":
                        liquibase_column(table, column)
            elif tag == "addColumn":
                table = get_table(model, change.get("tableName", ""))
                add_source(table, "liquibase", origin)
                for column in change:
                    if localname(column.tag) == "column":
                        liquibase_column(table, column)
            elif tag == "dropColumn":
                table = get_table(model, change.get("tableName", ""))
                cn = change.get("columnName")
                if cn and cn.lower() in table["cols"]:
                    del table["cols"][cn.lower()]
            elif tag == "addPrimaryKey":
                table = get_table(model, change.get("tableName", ""))
                for c in (change.get("columnNames", "")).split(","):
                    cn = c.strip()
                    if cn:
                        ex = table["cols"].get(cn.lower())
                        if ex:
                            ex["pk"] = True
                        if cn not in table["pk"]:
                            table["pk"].append(cn)
            elif tag == "addForeignKeyConstraint":
                table = get_table(model, change.get("baseTableName", ""))
                base_cols = [c.strip() for c in change.get("baseColumnNames", "").split(",")]
                ref_table = change.get("referencedTableName", "")
                ref_cols = [c.strip() for c in change.get("referencedColumnNames", "id").split(",")]
                table["fks"].append({
                    "columns": base_cols, "ref_table": ref_table,
                    "ref_columns": ref_cols, "inferred": False,
                })
                for bc in base_cols:
                    ex = table["cols"].get(bc.lower())
                    if ex and not ex["fk"]:
                        ex["fk"] = {"table": ref_table, "column": ref_cols[0] if ref_cols else "id"}
            elif tag == "sql":
                if change.text:
                    parse_sql_blob(model, change.text, "liquibase", origin)
    return True


def liquibase_column(table, column):
    col = new_col(column.get("name", ""))
    set_col_type(col, column.get("type", ""))
    if column.get("remarks"):
        col["remarks"] = column.get("remarks")
    default = (column.get("defaultValue") or column.get("defaultValueNumeric")
               or column.get("defaultValueComputed") or column.get("defaultValueBoolean"))
    if default is not None:
        col["default"] = str(default)
    if (column.get("autoIncrement") or "").lower() == "true":
        col["default"] = "auto-increment"
    for constraints in column:
        if localname(constraints.tag) != "constraints":
            continue
        if (constraints.get("primaryKey") or "").lower() == "true":
            col["pk"] = True
        if (constraints.get("nullable") or "").lower() == "false":
            col["nullable"] = False
        if (constraints.get("unique") or "").lower() == "true":
            col["unique"] = True
        ref = constraints.get("references")
        if ref:
            parsed = parse_references(ref)
            if parsed:
                col["fk"] = parsed
                table["fks"].append({
                    "columns": [col["name"]], "ref_table": parsed["table"],
                    "ref_columns": [parsed["column"]], "inferred": False,
                })
        elif constraints.get("referencedTableName"):
            rt = constraints.get("referencedTableName")
            rc = constraints.get("referencedColumnNames", "id")
            col["fk"] = {"table": rt, "column": rc}
            table["fks"].append({
                "columns": [col["name"]], "ref_table": rt,
                "ref_columns": [rc], "inferred": False,
            })
    upsert_col(table, col, "liquibase")


def parse_liquibase_json(model, path, rel):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
    except Exception:
        return False
    if not isinstance(data, dict) or "databaseChangeLog" not in data:
        return False
    for entry in data.get("databaseChangeLog", []):
        cs = entry.get("changeSet") if isinstance(entry, dict) else None
        if not cs:
            continue
        origin = "%s (changeSet %s)" % (rel, cs.get("id", ""))
        for change in cs.get("changes", []):
            ct = change.get("createTable")
            if ct:
                table = get_table(model, ct.get("tableName", ""))
                add_source(table, "liquibase", origin)
                if ct.get("remarks"):
                    table["remarks"] = ct["remarks"]
                for cw in ct.get("columns", []):
                    cdef = cw.get("column", cw)
                    col = new_col(cdef.get("name", ""))
                    set_col_type(col, cdef.get("type", ""))
                    if cdef.get("remarks"):
                        col["remarks"] = cdef["remarks"]
                    cons = cdef.get("constraints", {})
                    if cons.get("primaryKey"):
                        col["pk"] = True
                    if cons.get("nullable") is False:
                        col["nullable"] = False
                    if cons.get("references"):
                        parsed = parse_references(cons["references"])
                        if parsed:
                            col["fk"] = parsed
                            table["fks"].append({
                                "columns": [col["name"]], "ref_table": parsed["table"],
                                "ref_columns": [parsed["column"]], "inferred": False,
                            })
                    upsert_col(table, col, "liquibase")
    return True


def extract_java_strings(text):
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    blocks = []
    cleaned = []
    last = 0
    for m in re.finditer(r'"""(.*?)"""', text, re.DOTALL):
        blocks.append(m.group(1))
        cleaned.append(text[last:m.start()])
        last = m.end()
    cleaned.append(text[last:])
    remainder = " ".join(cleaned)
    literals = []
    for m in re.finditer(r'"((?:\\.|[^"\\])*)"', remainder):
        s = m.group(1)
        s = s.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
        literals.append(s)
    return blocks, literals


def parse_jdbc(model, text, rel, seeds):
    blocks, literals = extract_java_strings(text)
    candidates = list(blocks)
    candidates.append("\n".join(literals))
    for blob in candidates:
        if "CREATE TABLE" in blob.upper():
            parse_sql_blob(model, blob, "jdbc", "%s (JdbcTemplate)" % rel)
    for lit in literals + blocks:
        for stmt in re.finditer(r"INSERT\s+INTO\s+([\w\"`.]+)", lit, re.IGNORECASE):
            table_name = stmt.group(1).strip('"').strip("`").split(".")[-1]
            sql = lit.strip()
            if not sql.endswith(";"):
                sql = sql + ";"
            seeds.append({"table": table_name.lower(), "sql": sql})
            break


def parse_jpa(model, text, rel):
    if "@Entity" not in text:
        return
    text_nc = re.sub(r"//[^\n]*", "", text)
    text_nc = re.sub(r"/\*.*?\*/", "", text_nc, flags=re.DOTALL)
    if "@Entity" not in text_nc:
        return
    class_m = re.search(r"\bclass\s+(\w+)", text_nc)
    if not class_m:
        return
    class_name = class_m.group(1)
    table_m = re.search(r"@Table\s*\(([^)]*)\)", text_nc)
    table_name = None
    if table_m:
        nm = re.search(r'name\s*=\s*"([^"]+)"', table_m.group(1))
        if nm:
            table_name = nm.group(1)
    if not table_name:
        table_name = snake(class_name)
    table = get_table(model, table_name)
    add_source(table, "hibernate", "%s (@Entity %s)" % (rel, class_name))

    field_re = re.compile(
        r"(?P<ann>(?:@[\w.]+(?:\s*\([^)]*\))?\s*)*)"
        r"(?P<mods>(?:public|private|protected|static|final|transient|volatile)\s+)+"
        r"(?P<type>[\w.]+(?:\s*<[^>]*>)?(?:\[\])?)\s+"
        r"(?P<name>\w+)\s*(?:=\s*[^;]+)?;"
    )
    for m in field_re.finditer(text_nc):
        ann = m.group("ann") or ""
        mods = m.group("mods") or ""
        jtype = m.group("type").strip()
        fname = m.group("name")
        if "static" in mods:
            continue
        if "@Transient" in ann:
            continue
        col = new_col(fname)
        nm = re.search(r'@(?:Column|JoinColumn)\s*\([^)]*name\s*=\s*"([^"]+)"', ann)
        if nm:
            col["name"] = nm.group(1)
        else:
            col["name"] = snake(fname)

        length_m = re.search(r"length\s*=\s*(\d+)", ann)
        precision_m = re.search(r"precision\s*=\s*(\d+)", ann)
        scale_m = re.search(r"scale\s*=\s*(\d+)", ann)

        is_relation = bool(re.search(r"@(ManyToOne|OneToOne)", ann))
        if is_relation:
            set_col_type(col, "BIGINT")
            ref_table = snake(re.sub(r"<.*>", "", jtype))
            col["fk"] = {"table": ref_table, "column": "id"}
            table["fks"].append({
                "columns": [col["name"]], "ref_table": ref_table,
                "ref_columns": ["id"], "inferred": False,
            })
        elif re.search(r"@(OneToMany|ManyToMany)", ann):
            continue
        elif "@Enumerated" in ann:
            if "ORDINAL" in ann:
                set_col_type(col, "INT")
            else:
                set_col_type(col, "VARCHAR(%s)" % (length_m.group(1) if length_m else "255"))
        else:
            mapped = JAVA_TYPE_MAP.get(jtype)
            if mapped is None:
                bare = jtype.split(".")[-1]
                mapped = JAVA_TYPE_MAP.get(bare, "VARCHAR(255)")
            if mapped == "VARCHAR":
                mapped = "VARCHAR(%s)" % (length_m.group(1) if length_m else "255")
            elif mapped == "NUMERIC" and (precision_m or scale_m):
                mapped = "NUMERIC(%s,%s)" % (
                    precision_m.group(1) if precision_m else "19",
                    scale_m.group(1) if scale_m else "2")
            elif mapped == "NUMERIC":
                mapped = "NUMERIC(19,2)"
            set_col_type(col, mapped)

        if "@Id" in ann:
            col["pk"] = True
            col["nullable"] = False
        if "@GeneratedValue" in ann:
            col["default"] = "generated"
        if re.search(r"nullable\s*=\s*false", ann):
            col["nullable"] = False
        if re.search(r"unique\s*=\s*true", ann):
            col["unique"] = True
        upsert_col(table, col, "hibernate")


def infer_fks(model):
    names = set(model.keys())

    def find_target(prefix):
        p = prefix.lower()
        for cand in (p, p[:-1] if p.endswith("s") else None, p + "s"):
            if cand and cand in names:
                return cand
        return None

    for table in model.values():
        declared = set()
        for fk in table["fks"]:
            for c in fk["columns"]:
                declared.add(c.lower())
        for col in table["cols"].values():
            cname = col["name"].lower()
            if cname in declared or col["fk"]:
                continue
            m = re.match(r"^(.*)_id$", cname)
            if not m:
                continue
            target = find_target(m.group(1))
            if not target or target == table["name"].lower():
                continue
            col["fk"] = {"table": model[target]["name"], "column": "id", "inferred": True}
            table["fks"].append({
                "columns": [col["name"]], "ref_table": model[target]["name"],
                "ref_columns": ["id"], "inferred": True,
            })


def est_row_bytes(table):
    total = 0
    for col in table["cols"].values():
        base = col["base"]
        if base in ("VARCHAR", "CHAR", "CHARACTER", "NVARCHAR", "VARCHAR2"):
            total += col["size"] or 255
        elif base in ("NUMERIC", "DECIMAL", "NUMBER"):
            total += (col["precision"] or 18) // 2 + 1
        else:
            total += TYPE_WIDTH.get(base, 8)
    return total


SQLITE_TEXT = {"VARCHAR", "CHAR", "CHARACTER", "NVARCHAR", "VARCHAR2", "TEXT",
               "CLOB", "DATE", "TIME", "TIMESTAMP", "DATETIME", "UUID", "ENUM"}
SQLITE_INT = {"BIGINT", "INT", "INTEGER", "SMALLINT", "TINYINT", "BOOLEAN", "BOOL"}
SQLITE_REAL = {"DOUBLE", "FLOAT", "REAL"}


def sqlite_type(col):
    base = col["base"]
    if base in SQLITE_INT:
        return "INTEGER"
    if base in SQLITE_REAL:
        return "REAL"
    if base in ("NUMERIC", "DECIMAL", "NUMBER"):
        return "NUMERIC"
    if base == "BLOB":
        return "BLOB"
    return "TEXT"


def build_sqlite_ddl(tables):
    out = []
    for table in tables:
        cols = list(table["cols"].values())
        if not cols:
            continue
        single_pk = table["pk"][0] if len(table["pk"]) == 1 else None
        lines = []
        for col in cols:
            t = sqlite_type(col)
            line = '  "%s" %s' % (col["name"], t)
            if single_pk and col["name"] == single_pk and t == "INTEGER":
                line += " PRIMARY KEY"
            if col["default"] and col["default"] not in ("auto-increment", "generated"):
                dv = col["default"]
                if dv.upper() == "CURRENT_TIMESTAMP":
                    line += " DEFAULT CURRENT_TIMESTAMP"
                elif re.match(r"^-?\d+(\.\d+)?$", dv):
                    line += " DEFAULT %s" % dv
                else:
                    line += " DEFAULT '%s'" % dv.replace("'", "''")
            lines.append(line)
        if len(table["pk"]) > 1:
            lines.append('  PRIMARY KEY (%s)' % ", ".join('"%s"' % c for c in table["pk"]))
        out.append('CREATE TABLE "%s" (\n%s\n);' % (table["name"], ",\n".join(lines)))
    return "\n".join(out)


def col_description(col):
    if col["remarks"]:
        return col["remarks"]
    if col["pk"] and col["name"].lower() == "id":
        return "Surrogate primary key"
    if col["fk"]:
        return "Reference to %s" % col["fk"]["table"]
    if col["name"].lower() == "id":
        return "Primary key"
    return col["name"].replace("_", " ").strip().capitalize()


def table_description(table):
    if table["remarks"]:
        return table["remarks"]
    return table["name"].replace("_", " ").strip().capitalize()


def source_label(sources):
    if not sources:
        return "unknown"
    if len(sources) > 1:
        return "mixed"
    return sources[0]


def walk_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:
        return ""


def main():
    scan = sys.argv[1] if len(sys.argv) > 1 else "."
    scan = os.path.abspath(scan)
    if not os.path.exists(scan):
        print("path not found: %s" % scan)
        sys.exit(1)

    model = OrderedDict()
    seeds = []

    for path in walk_files(scan):
        rel = os.path.relpath(path, scan)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".xml":
            parse_liquibase_xml(model, path, rel)
        elif ext == ".json":
            text = read_text(path)
            if "databaseChangeLog" in text:
                parse_liquibase_json(model, path, rel)
        elif ext == ".sql":
            text = read_text(path)
            src = "liquibase" if "liquibase formatted sql" in text.lower() else "sql"
            parse_sql_blob(model, text, src, rel)
            for stmt in re.finditer(r"INSERT\s+INTO\s+([\w\"`.]+)[^;]*;", text, re.IGNORECASE):
                tname = stmt.group(1).strip('"').strip("`").split(".")[-1]
                seeds.append({"table": tname.lower(), "sql": stmt.group(0).strip()})
        elif ext == ".java":
            text = read_text(path)
            if "@Entity" in text:
                parse_jpa(model, text, rel)
            if "CREATE TABLE" in text.upper() or "INSERT INTO" in text.upper():
                parse_jdbc(model, text, rel, seeds)

    infer_fks(model)

    tables = [t for t in model.values() if t["cols"]]
    tables.sort(key=lambda t: t["name"])

    out_tables = []
    relationships = []
    tally = {}
    total_cols = 0
    seed_counts = {}
    for s in seeds:
        seed_counts[s["table"]] = seed_counts.get(s["table"], 0) + 1

    for table in tables:
        label = source_label(table["sources"])
        tally[label] = tally.get(label, 0) + 1
        cols_out = []
        for col in table["cols"].values():
            total_cols += 1
            inferred_fk = bool(col["fk"] and col["fk"].get("inferred"))
            cols_out.append({
                "name": col["name"],
                "type": col["type"] or col["base"],
                "size": col["size"],
                "nullable": col["nullable"],
                "pk": col["pk"],
                "unique": col["unique"],
                "default": col["default"],
                "fk": col["fk"],
                "fk_inferred": inferred_fk,
                "source": col["source"],
                "description": col_description(col),
            })
        for fk in table["fks"]:
            relationships.append({
                "from": table["name"],
                "to": fk["ref_table"],
                "from_cols": fk["columns"],
                "to_cols": fk["ref_columns"],
                "inferred": fk.get("inferred", False),
            })
        out_tables.append({
            "name": table["name"],
            "source": label,
            "sources": table["sources"],
            "origins": table["origins"],
            "description": table_description(table),
            "remarks": table["remarks"],
            "columns": cols_out,
            "column_count": len(cols_out),
            "pk": table["pk"],
            "est_row_bytes": est_row_bytes(table),
            "seed_rows": seed_counts.get(table["name"].lower(), 0),
        })

    valid_tables = {t["name"].lower() for t in tables}
    relationships = [r for r in relationships if r["to"].lower() in valid_tables]
    ordered_seeds = [s["sql"] for s in seeds if s["table"] in valid_tables]

    data = {
        "project": os.path.basename(scan.rstrip("/")) or scan,
        "scan_path": scan,
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "table_count": len(out_tables),
        "column_count": total_cols,
        "tally": tally,
        "tables": out_tables,
        "relationships": relationships,
        "sqlite_ddl": build_sqlite_ddl(tables),
        "seeds": ordered_seeds,
    }

    here = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(here, "..", "assets", "template.html")
    out_dir = os.path.join(os.getcwd(), "data-dict-report")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "data.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    with open(template_path, "r", encoding="utf-8") as fh:
        template = fh.read()
    payload = json.dumps(data).replace("</", "<\\/")
    rendered = template.replace("__DATADICT_DATA__", payload)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(rendered)

    print("Data dictionary for: %s" % data["project"])
    print("Tables discovered: %d (%d columns)" % (data["table_count"], data["column_count"]))
    for src in sorted(tally):
        print("  %s: %d" % (src, tally[src]))
    print("Relationships: %d (%d inferred)" % (
        len(relationships), sum(1 for r in relationships if r["inferred"])))
    print("Report: %s" % os.path.join(out_dir, "index.html"))


if __name__ == "__main__":
    main()

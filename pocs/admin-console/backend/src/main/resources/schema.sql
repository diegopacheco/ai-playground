CREATE TABLE IF NOT EXISTS keys (
    id BIGSERIAL PRIMARY KEY,
    purpose VARCHAR(32) NOT NULL UNIQUE,
    key_material BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    password_hash BYTEA NOT NULL,
    password_salt BYTEA NOT NULL,
    role VARCHAR(16) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS projects (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(128) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by VARCHAR(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS connections (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(128) NOT NULL,
    kind VARCHAR(32) NOT NULL,
    host VARCHAR(255) NOT NULL,
    port INTEGER NOT NULL,
    database VARCHAR(128),
    keyspace VARCHAR(128),
    datacenter VARCHAR(128),
    username VARCHAR(128),
    secret_ciphertext BYTEA,
    options_json TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by VARCHAR(64) NOT NULL,
    UNIQUE (project_id, name)
);

CREATE TABLE IF NOT EXISTS saved_queries (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    connection_id BIGINT REFERENCES connections(id) ON DELETE SET NULL,
    name VARCHAR(128) NOT NULL,
    statement TEXT NOT NULL,
    kind VARCHAR(32) NOT NULL,
    description TEXT,
    created_by VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (project_id, name)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    query_id UUID NOT NULL,
    page INTEGER NOT NULL,
    at TIMESTAMPTZ NOT NULL DEFAULT now(),
    username VARCHAR(64) NOT NULL,
    connection_id BIGINT,
    project_id BIGINT,
    kind VARCHAR(32),
    statement TEXT NOT NULL,
    allowed BOOLEAN NOT NULL,
    denial_reason TEXT,
    elapsed_ms BIGINT,
    row_count INTEGER,
    error TEXT,
    client_ip VARCHAR(64)
);

CREATE INDEX IF NOT EXISTS audit_log_at_idx ON audit_log (at DESC);
CREATE INDEX IF NOT EXISTS audit_log_user_at_idx ON audit_log (username, at DESC);
CREATE INDEX IF NOT EXISTS audit_log_connection_at_idx ON audit_log (connection_id, at DESC);
CREATE INDEX IF NOT EXISTS audit_log_query_idx ON audit_log (query_id, page);

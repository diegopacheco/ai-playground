CREATE ROLE agentinfra_app LOGIN PASSWORD 'agentinfra_app';

CREATE TABLE kv (
  tenant TEXT NOT NULL,
  key TEXT NOT NULL,
  value JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant, key)
);

CREATE TABLE eval_runs (
  id BIGSERIAL PRIMARY KEY,
  tenant TEXT NOT NULL,
  suite TEXT NOT NULL,
  passed INT NOT NULL,
  failed INT NOT NULL,
  score NUMERIC(5,4) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE eval_cases (
  id BIGSERIAL PRIMARY KEY,
  run_id BIGINT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
  tenant TEXT NOT NULL,
  name TEXT NOT NULL,
  input TEXT NOT NULL,
  output TEXT NOT NULL,
  latency_ms INT NOT NULL,
  passed BOOLEAN NOT NULL,
  checks JSONB NOT NULL
);

CREATE INDEX eval_runs_tenant_suite ON eval_runs (tenant, suite, created_at DESC);
CREATE INDEX eval_cases_run ON eval_cases (run_id);

ALTER TABLE kv ENABLE ROW LEVEL SECURITY;
ALTER TABLE kv FORCE ROW LEVEL SECURITY;
ALTER TABLE eval_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE eval_runs FORCE ROW LEVEL SECURITY;
ALTER TABLE eval_cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE eval_cases FORCE ROW LEVEL SECURITY;

CREATE POLICY kv_tenant ON kv
  USING (tenant = current_setting('app.tenant', true))
  WITH CHECK (tenant = current_setting('app.tenant', true));

CREATE POLICY eval_runs_tenant ON eval_runs
  USING (tenant = current_setting('app.tenant', true) OR current_setting('app.read_all', true) = 'on')
  WITH CHECK (tenant = current_setting('app.tenant', true));

CREATE POLICY eval_cases_tenant ON eval_cases
  USING (tenant = current_setting('app.tenant', true) OR current_setting('app.read_all', true) = 'on')
  WITH CHECK (tenant = current_setting('app.tenant', true));

GRANT SELECT, INSERT, UPDATE, DELETE ON kv, eval_runs, eval_cases TO agentinfra_app;
GRANT USAGE, SELECT ON SEQUENCE eval_runs_id_seq, eval_cases_id_seq TO agentinfra_app;

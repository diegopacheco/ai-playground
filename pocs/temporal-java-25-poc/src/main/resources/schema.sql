CREATE TABLE IF NOT EXISTS research_report (
  id BIGSERIAL PRIMARY KEY,
  symbol VARCHAR(16) NOT NULL,
  company VARCHAR(160) NOT NULL,
  stock_summary TEXT NOT NULL,
  news_summary TEXT NOT NULL,
  recommendation VARCHAR(16) NOT NULL,
  confidence INTEGER NOT NULL,
  rationale TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

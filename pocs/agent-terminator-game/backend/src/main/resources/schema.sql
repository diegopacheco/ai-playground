CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,
    terminator_agent TEXT NOT NULL,
    terminator_model TEXT NOT NULL,
    mosquito_agent TEXT NOT NULL,
    mosquito_model TEXT NOT NULL,
    grid_size INTEGER NOT NULL DEFAULT 20,
    winner TEXT,
    total_cycles INTEGER DEFAULT 0,
    max_mosquitos INTEGER DEFAULT 0,
    total_kills INTEGER DEFAULT 0,
    total_hatched INTEGER DEFAULT 0,
    total_dates INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS game_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(id)
);

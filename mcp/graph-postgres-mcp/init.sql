CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    published BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

INSERT INTO users (name, email) VALUES
    ('Alice', 'alice@test.com'),
    ('Bob', 'bob@test.com');

INSERT INTO posts (user_id, title, body, published) VALUES
    (1, 'First Post', 'Hello World', true),
    (1, 'Second Post', 'GraphQL is great', false),
    (2, 'Bobs Post', 'Testing MCP', true);

INSERT INTO tags (name) VALUES ('tech'), ('graphql'), ('mcp');

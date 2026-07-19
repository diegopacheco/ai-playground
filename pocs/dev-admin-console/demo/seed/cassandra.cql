CREATE KEYSPACE IF NOT EXISTS shop
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE shop;

DROP TABLE IF EXISTS events_by_customer;
DROP TABLE IF EXISTS sessions;

CREATE TABLE events_by_customer (
    customer_id int,
    event_time timestamp,
    event_id uuid,
    event_type text,
    payload text,
    PRIMARY KEY ((customer_id), event_time, event_id)
) WITH CLUSTERING ORDER BY (event_time DESC, event_id ASC);

CREATE TABLE sessions (
    session_id uuid PRIMARY KEY,
    customer_id int,
    started_at timestamp,
    user_agent text
);

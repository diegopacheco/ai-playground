CREATE TABLE retweets (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tweet_id INTEGER NOT NULL REFERENCES tweets(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, tweet_id)
);

CREATE INDEX idx_retweets_user_id ON retweets(user_id);
CREATE INDEX idx_retweets_tweet_id ON retweets(tweet_id);

use crate::redis::{Redis, Reply};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub key: String,
    pub question: String,
    pub answer: String,
    pub similarity: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub key: String,
    pub question: String,
    pub answer: String,
    pub hits: i64,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stats {
    pub hits: i64,
    pub misses: i64,
    pub entries: i64,
}

pub trait Store: Send + Sync {
    fn nearest(&self, vector: &[f32]) -> Result<Option<Hit>, String>;
    fn save(&self, question: &str, answer: &str, vector: &[f32]) -> Result<(), String>;
    fn record_hit(&self, key: &str) -> Result<(), String>;
    fn record_miss(&self) -> Result<(), String>;
    fn entries(&self) -> Result<Vec<Entry>, String>;
    fn stats(&self) -> Result<Stats, String>;
    fn clear(&self) -> Result<(), String>;
}

pub struct RedisStore {
    redis: Redis,
    dimensions: usize,
    namespace: String,
}

impl RedisStore {
    pub fn open(redis: Redis, dimensions: usize, namespace: &str) -> Result<Self, String> {
        let store = Self { redis, dimensions, namespace: namespace.to_string() };
        store.create_index()?;
        Ok(store)
    }

    fn index(&self) -> String {
        format!("{}_idx", self.namespace)
    }

    fn counter(&self, name: &str) -> String {
        format!("{}_stats:{name}", self.namespace)
    }

    fn create_index(&self) -> Result<(), String> {
        let prefix = format!("{}:", self.namespace);
        let dimensions = self.dimensions.to_string();
        let result = self.redis.words(&[
            "FT.CREATE", &self.index(), "ON", "HASH", "PREFIX", "1", &prefix, "SCHEMA",
            "question", "TEXT", "answer", "TEXT", "created_at", "NUMERIC", "SORTABLE", "hits", "NUMERIC",
            "embedding", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", &dimensions, "DISTANCE_METRIC", "COSINE",
        ]);
        match result {
            Err(message) if !message.contains("already exists") => Err(message),
            _ => Ok(()),
        }
    }
}

impl Store for RedisStore {
    fn nearest(&self, vector: &[f32]) -> Result<Option<Hit>, String> {
        let index = self.index();
        let blob = to_blob(vector);
        let reply = self.redis.command(&[
            b"FT.SEARCH", index.as_bytes(), b"*=>[KNN 1 @embedding $vec AS distance]", b"PARAMS", b"2", b"vec", &blob,
            b"SORTBY", b"distance", b"RETURN", b"3", b"question", b"answer", b"distance", b"DIALECT", b"2",
        ])?;
        Ok(documents(&reply).into_iter().next().map(|(key, fields)| Hit {
            key,
            question: field(&fields, "question"),
            answer: field(&fields, "answer"),
            similarity: 1.0 - field(&fields, "distance").parse::<f32>().unwrap_or(1.0),
        }))
    }

    fn save(&self, question: &str, answer: &str, vector: &[f32]) -> Result<(), String> {
        let created_at = now_millis();
        let key = format!("{}:{}", self.namespace, now_nanos());
        let blob = to_blob(vector);
        let created = created_at.to_string();
        self.redis
            .command(&[b"HSET", key.as_bytes(), b"question", question.as_bytes(), b"answer", answer.as_bytes(), b"created_at", created.as_bytes(), b"hits", b"0", b"embedding", &blob])
            .map(|_| ())
    }

    fn record_hit(&self, key: &str) -> Result<(), String> {
        self.redis.words(&["HINCRBY", key, "hits", "1"])?;
        self.redis.words(&["INCR", &self.counter("hits")]).map(|_| ())
    }

    fn record_miss(&self) -> Result<(), String> {
        self.redis.words(&["INCR", &self.counter("misses")]).map(|_| ())
    }

    fn entries(&self) -> Result<Vec<Entry>, String> {
        let reply = self.redis.words(&[
            "FT.SEARCH", &self.index(), "*", "SORTBY", "created_at", "DESC", "RETURN", "4", "question", "answer", "hits", "created_at", "LIMIT", "0", "100", "DIALECT", "2",
        ])?;
        Ok(documents(&reply)
            .into_iter()
            .map(|(key, fields)| Entry {
                key,
                question: field(&fields, "question"),
                answer: field(&fields, "answer"),
                hits: field(&fields, "hits").parse().unwrap_or(0),
                created_at: field(&fields, "created_at").parse().unwrap_or(0),
            })
            .collect())
    }

    fn stats(&self) -> Result<Stats, String> {
        let count = |name: &str| -> Result<i64, String> { Ok(self.redis.words(&["GET", &self.counter(name)])?.integer().unwrap_or(0)) };
        let total = self.redis.words(&["FT.SEARCH", &self.index(), "*", "LIMIT", "0", "0", "DIALECT", "2"])?;
        Ok(Stats { hits: count("hits")?, misses: count("misses")?, entries: total.items().first().and_then(Reply::integer).unwrap_or(0) })
    }

    fn clear(&self) -> Result<(), String> {
        self.redis.words(&["FT.DROPINDEX", &self.index(), "DD"])?;
        self.redis.words(&["DEL", &self.counter("hits"), &self.counter("misses")])?;
        self.create_index()
    }
}

pub fn to_blob(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn documents(reply: &Reply) -> Vec<(String, Vec<(String, String)>)> {
    let items = reply.items();
    if items.is_empty() {
        return Vec::new();
    }
    items[1..]
        .chunks(2)
        .filter_map(|pair| {
            let key = pair[0].text()?;
            let fields = pair.get(1)?.items().chunks(2).filter_map(|kv| Some((kv[0].text()?, kv.get(1)?.text()?))).collect();
            Some((key, fields))
        })
        .collect()
}

fn field(fields: &[(String, String)], name: &str) -> String {
    fields.iter().find(|(key, _)| key == name).map(|(_, value)| value.clone()).unwrap_or_default()
}

fn now_millis() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|value| value.as_millis() as i64).unwrap_or(0)
}

fn now_nanos() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|value| value.as_nanos()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_are_little_endian_float32_because_that_is_what_the_hnsw_index_reads() {
        assert_eq!(to_blob(&[1.0]), vec![0, 0, 0x80, 0x3f]);
        assert_eq!(to_blob(&[1.0, 2.0]).len(), 8);
    }

    #[test]
    fn search_documents_keep_their_key_and_fields() {
        let bulk = |value: &str| Reply::Bulk(Some(value.as_bytes().to_vec()));
        let reply = Reply::Array(vec![Reply::Integer(1), bulk("qa:1"), Reply::Array(vec![bulk("question"), bulk("why?"), bulk("distance"), bulk("0.25")])]);
        let docs = documents(&reply);
        assert_eq!(docs[0].0, "qa:1");
        assert_eq!(field(&docs[0].1, "distance"), "0.25");
        assert_eq!(field(&docs[0].1, "absent"), "");
    }
}

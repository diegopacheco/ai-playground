use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct Follow {
    pub id: i64,
    pub follower_id: i64,
    pub following_id: i64,
    pub created_at: String,
}

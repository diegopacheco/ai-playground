use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct User {
    pub id: Uuid,
    pub name: String,
    pub email: String,
    pub created_at: NaiveDateTime,
}

#[derive(Debug, Deserialize)]
pub struct CreateUser {
    pub name: String,
    pub email: String,
}

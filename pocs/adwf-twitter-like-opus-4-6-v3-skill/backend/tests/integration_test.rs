use reqwest::Client;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU16, Ordering};

static PORT_COUNTER: AtomicU16 = AtomicU16::new(4000);

fn next_port() -> u16 {
    PORT_COUNTER.fetch_add(1, Ordering::SeqCst)
}

async fn start_server() -> (String, u16) {
    let port = next_port();
    let db_path = format!("/tmp/test_twitter_{}.db", port);
    let _ = std::fs::remove_file(&db_path);

    let pool = twitter_backend::create_test_pool(&db_path).await;
    let app = twitter_backend::build_app(pool);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port))
        .await
        .expect("Failed to bind test port");

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let base_url = format!("http://127.0.0.1:{}", port);
    (base_url, port)
}

async fn register_user(client: &Client, base: &str, username: &str, email: &str, password: &str) -> Value {
    let res = client
        .post(format!("{}/api/auth/register", base))
        .json(&json!({
            "username": username,
            "email": email,
            "password": password
        }))
        .send()
        .await
        .unwrap();
    res.json::<Value>().await.unwrap()
}

async fn login_user(client: &Client, base: &str, email: &str, password: &str) -> Value {
    let res = client
        .post(format!("{}/api/auth/login", base))
        .json(&json!({
            "email": email,
            "password": password
        }))
        .send()
        .await
        .unwrap();
    res.json::<Value>().await.unwrap()
}

#[tokio::test]
async fn test_register_flow() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let data = register_user(&client, &base, "alice", "alice@test.com", "pass123").await;

    assert!(data["token"].is_string());
    assert_eq!(data["user"]["username"], "alice");
    assert_eq!(data["user"]["email"], "alice@test.com");
    assert!(data["user"]["id"].is_string());
}

#[tokio::test]
async fn test_login_flow() {
    let (base, _) = start_server().await;
    let client = Client::new();

    register_user(&client, &base, "bob", "bob@test.com", "mypass").await;
    let data = login_user(&client, &base, "bob@test.com", "mypass").await;

    assert!(data["token"].is_string());
    assert_eq!(data["user"]["username"], "bob");
}

#[tokio::test]
async fn test_login_wrong_password() {
    let (base, _) = start_server().await;
    let client = Client::new();

    register_user(&client, &base, "charlie", "charlie@test.com", "correct").await;

    let res = client
        .post(format!("{}/api/auth/login", base))
        .json(&json!({
            "email": "charlie@test.com",
            "password": "wrong"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 401);
}

#[tokio::test]
async fn test_create_post() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth = register_user(&client, &base, "dave", "dave@test.com", "pass").await;
    let token = auth["token"].as_str().unwrap();

    let res = client
        .post(format!("{}/api/posts", base))
        .bearer_auth(token)
        .json(&json!({ "content": "Hello world!" }))
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 200);
    let post: Value = res.json().await.unwrap();
    assert_eq!(post["content"], "Hello world!");
    assert_eq!(post["username"], "dave");
    assert_eq!(post["likes_count"], 0);
    assert_eq!(post["liked_by_me"], false);
}

#[tokio::test]
async fn test_get_timeline() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth = register_user(&client, &base, "eve", "eve@test.com", "pass").await;
    let token = auth["token"].as_str().unwrap();

    client
        .post(format!("{}/api/posts", base))
        .bearer_auth(token)
        .json(&json!({ "content": "Post 1" }))
        .send()
        .await
        .unwrap();

    client
        .post(format!("{}/api/posts", base))
        .bearer_auth(token)
        .json(&json!({ "content": "Post 2" }))
        .send()
        .await
        .unwrap();

    let res = client
        .get(format!("{}/api/posts", base))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 200);
    let posts: Vec<Value> = res.json().await.unwrap();
    assert_eq!(posts.len(), 2);
}

#[tokio::test]
async fn test_like_unlike() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth = register_user(&client, &base, "frank", "frank@test.com", "pass").await;
    let token = auth["token"].as_str().unwrap();

    let post_res = client
        .post(format!("{}/api/posts", base))
        .bearer_auth(token)
        .json(&json!({ "content": "Like me" }))
        .send()
        .await
        .unwrap();
    let post: Value = post_res.json().await.unwrap();
    let post_id = post["id"].as_str().unwrap();

    let like_res = client
        .post(format!("{}/api/posts/{}/like", base, post_id))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    assert_eq!(like_res.status(), 200);

    let get_res = client
        .get(format!("{}/api/posts/{}", base, post_id))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    let liked_post: Value = get_res.json().await.unwrap();
    assert_eq!(liked_post["likes_count"], 1);
    assert_eq!(liked_post["liked_by_me"], true);

    let unlike_res = client
        .delete(format!("{}/api/posts/{}/like", base, post_id))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    assert_eq!(unlike_res.status(), 200);

    let get_res2 = client
        .get(format!("{}/api/posts/{}", base, post_id))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    let unliked_post: Value = get_res2.json().await.unwrap();
    assert_eq!(unliked_post["likes_count"], 0);
    assert_eq!(unliked_post["liked_by_me"], false);
}

#[tokio::test]
async fn test_follow_unfollow() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth1 = register_user(&client, &base, "grace", "grace@test.com", "pass").await;
    let token1 = auth1["token"].as_str().unwrap();

    let auth2 = register_user(&client, &base, "henry", "henry@test.com", "pass").await;
    let user2_id = auth2["user"]["id"].as_str().unwrap();

    let follow_res = client
        .post(format!("{}/api/users/{}/follow", base, user2_id))
        .bearer_auth(token1)
        .send()
        .await
        .unwrap();
    assert_eq!(follow_res.status(), 200);

    let profile_res = client
        .get(format!("{}/api/users/{}", base, user2_id))
        .bearer_auth(token1)
        .send()
        .await
        .unwrap();
    let profile: Value = profile_res.json().await.unwrap();
    assert_eq!(profile["followers_count"], 1);
    assert_eq!(profile["is_following"], true);

    let unfollow_res = client
        .delete(format!("{}/api/users/{}/follow", base, user2_id))
        .bearer_auth(token1)
        .send()
        .await
        .unwrap();
    assert_eq!(unfollow_res.status(), 200);

    let profile_res2 = client
        .get(format!("{}/api/users/{}", base, user2_id))
        .bearer_auth(token1)
        .send()
        .await
        .unwrap();
    let profile2: Value = profile_res2.json().await.unwrap();
    assert_eq!(profile2["followers_count"], 0);
    assert_eq!(profile2["is_following"], false);
}

#[tokio::test]
async fn test_get_user_profile() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth = register_user(&client, &base, "iris", "iris@test.com", "pass").await;
    let token = auth["token"].as_str().unwrap();
    let user_id = auth["user"]["id"].as_str().unwrap();

    let res = client
        .get(format!("{}/api/users/{}", base, user_id))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 200);
    let profile: Value = res.json().await.unwrap();
    assert_eq!(profile["user"]["username"], "iris");
    assert_eq!(profile["followers_count"], 0);
    assert_eq!(profile["following_count"], 0);
    assert_eq!(profile["is_following"], false);
}

#[tokio::test]
async fn test_empty_content_validation() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth = register_user(&client, &base, "jack", "jack@test.com", "pass").await;
    let token = auth["token"].as_str().unwrap();

    let res = client
        .post(format!("{}/api/posts", base))
        .bearer_auth(token)
        .json(&json!({ "content": "" }))
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 400);
}

#[tokio::test]
async fn test_too_long_content_validation() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth = register_user(&client, &base, "kate", "kate@test.com", "pass").await;
    let token = auth["token"].as_str().unwrap();

    let long_content = "a".repeat(281);

    let res = client
        .post(format!("{}/api/posts", base))
        .bearer_auth(token)
        .json(&json!({ "content": long_content }))
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 400);
}

#[tokio::test]
async fn test_invalid_email_validation() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let res = client
        .post(format!("{}/api/auth/register", base))
        .json(&json!({
            "username": "leo",
            "email": "notemail",
            "password": "pass"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 400);
}

#[tokio::test]
async fn test_duplicate_register() {
    let (base, _) = start_server().await;
    let client = Client::new();

    register_user(&client, &base, "mike", "mike@test.com", "pass").await;

    let res = client
        .post(format!("{}/api/auth/register", base))
        .json(&json!({
            "username": "mike",
            "email": "mike@test.com",
            "password": "pass"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 409);
}

#[tokio::test]
async fn test_delete_post() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth = register_user(&client, &base, "nina", "nina@test.com", "pass").await;
    let token = auth["token"].as_str().unwrap();

    let post_res = client
        .post(format!("{}/api/posts", base))
        .bearer_auth(token)
        .json(&json!({ "content": "To be deleted" }))
        .send()
        .await
        .unwrap();
    let post: Value = post_res.json().await.unwrap();
    let post_id = post["id"].as_str().unwrap();

    let del_res = client
        .delete(format!("{}/api/posts/{}", base, post_id))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    assert_eq!(del_res.status(), 200);

    let get_res = client
        .get(format!("{}/api/posts/{}", base, post_id))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    assert_eq!(get_res.status(), 404);
}

#[tokio::test]
async fn test_get_user_posts() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth = register_user(&client, &base, "oscar", "oscar@test.com", "pass").await;
    let token = auth["token"].as_str().unwrap();
    let user_id = auth["user"]["id"].as_str().unwrap();

    client
        .post(format!("{}/api/posts", base))
        .bearer_auth(token)
        .json(&json!({ "content": "Oscar post 1" }))
        .send()
        .await
        .unwrap();

    let res = client
        .get(format!("{}/api/users/{}/posts", base, user_id))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 200);
    let posts: Vec<Value> = res.json().await.unwrap();
    assert_eq!(posts.len(), 1);
    assert_eq!(posts[0]["content"], "Oscar post 1");
}

#[tokio::test]
async fn test_get_followers_following() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth1 = register_user(&client, &base, "pat", "pat@test.com", "pass").await;
    let token1 = auth1["token"].as_str().unwrap();
    let user1_id = auth1["user"]["id"].as_str().unwrap();

    let auth2 = register_user(&client, &base, "quinn", "quinn@test.com", "pass").await;
    let token2 = auth2["token"].as_str().unwrap();
    let user2_id = auth2["user"]["id"].as_str().unwrap();

    client
        .post(format!("{}/api/users/{}/follow", base, user2_id))
        .bearer_auth(token1)
        .send()
        .await
        .unwrap();

    let followers_res = client
        .get(format!("{}/api/users/{}/followers", base, user2_id))
        .bearer_auth(token2)
        .send()
        .await
        .unwrap();
    let followers: Vec<Value> = followers_res.json().await.unwrap();
    assert_eq!(followers.len(), 1);
    assert_eq!(followers[0]["username"], "pat");

    let following_res = client
        .get(format!("{}/api/users/{}/following", base, user1_id))
        .bearer_auth(token1)
        .send()
        .await
        .unwrap();
    let following: Vec<Value> = following_res.json().await.unwrap();
    assert_eq!(following.len(), 1);
    assert_eq!(following[0]["username"], "quinn");
}

#[tokio::test]
async fn test_cannot_follow_self() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth = register_user(&client, &base, "ron", "ron@test.com", "pass").await;
    let token = auth["token"].as_str().unwrap();
    let user_id = auth["user"]["id"].as_str().unwrap();

    let res = client
        .post(format!("{}/api/users/{}/follow", base, user_id))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 400);
}

#[tokio::test]
async fn test_me_endpoint() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth = register_user(&client, &base, "sue", "sue@test.com", "pass").await;
    let token = auth["token"].as_str().unwrap();

    let res = client
        .get(format!("{}/api/auth/me", base))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 200);
    let user: Value = res.json().await.unwrap();
    assert_eq!(user["username"], "sue");
    assert_eq!(user["email"], "sue@test.com");
}

#[tokio::test]
async fn test_unauthorized_without_token() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let res = client
        .get(format!("{}/api/posts", base))
        .send()
        .await
        .unwrap();

    assert_eq!(res.status(), 401);
}

#[tokio::test]
async fn test_timeline_shows_followed_user_posts() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth1 = register_user(&client, &base, "tina", "tina@test.com", "pass").await;
    let token1 = auth1["token"].as_str().unwrap();

    let auth2 = register_user(&client, &base, "uma", "uma@test.com", "pass").await;
    let token2 = auth2["token"].as_str().unwrap();
    let user2_id = auth2["user"]["id"].as_str().unwrap();

    client
        .post(format!("{}/api/posts", base))
        .bearer_auth(token2)
        .json(&json!({ "content": "Uma's post" }))
        .send()
        .await
        .unwrap();

    client
        .post(format!("{}/api/users/{}/follow", base, user2_id))
        .bearer_auth(token1)
        .send()
        .await
        .unwrap();

    let timeline_res = client
        .get(format!("{}/api/posts", base))
        .bearer_auth(token1)
        .send()
        .await
        .unwrap();
    let posts: Vec<Value> = timeline_res.json().await.unwrap();
    assert_eq!(posts.len(), 1);
    assert_eq!(posts[0]["content"], "Uma's post");
}

#[tokio::test]
async fn test_double_like_conflict() {
    let (base, _) = start_server().await;
    let client = Client::new();

    let auth = register_user(&client, &base, "vera", "vera@test.com", "pass").await;
    let token = auth["token"].as_str().unwrap();

    let post_res = client
        .post(format!("{}/api/posts", base))
        .bearer_auth(token)
        .json(&json!({ "content": "Double like test" }))
        .send()
        .await
        .unwrap();
    let post: Value = post_res.json().await.unwrap();
    let post_id = post["id"].as_str().unwrap();

    client
        .post(format!("{}/api/posts/{}/like", base, post_id))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();

    let res = client
        .post(format!("{}/api/posts/{}/like", base, post_id))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    assert_eq!(res.status(), 409);
}

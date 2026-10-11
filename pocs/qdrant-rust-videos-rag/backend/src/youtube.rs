use anyhow::{Context, Result};
use reqwest::header::{ACCEPT_LANGUAGE, COOKIE, HeaderMap, HeaderValue, USER_AGENT};
use serde_json::{Value, json};
use tokio::task::JoinSet;

const CHANNEL_URL: &str = "https://www.youtube.com/channel/UC81qdxTnQWQxw73_CmIZAjg/videos";
const BROWSE_URL: &str = "https://www.youtube.com/youtubei/v1/browse?prettyPrint=false";
const PLAYER_URL: &str = "https://www.youtube.com/youtubei/v1/player?prettyPrint=false";
const BROWSER: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36";
const CONCURRENCY: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct Video {
    pub id: String,
    pub title: String,
    pub description: String,
    pub keywords: Vec<String>,
    pub duration_text: String,
    pub duration_seconds: u64,
    pub views: u64,
    pub age_text: String,
}

#[derive(Debug, Default, PartialEq)]
pub struct Details {
    pub description: String,
    pub keywords: Vec<String>,
    pub duration_seconds: Option<u64>,
    pub views: Option<u64>,
}

pub fn between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let from = text.find(start)? + start.len();
    let to = text[from..].find(end)? + from;
    Some(&text[from..to])
}

pub fn client_version(html: &str) -> Option<String> {
    between(html, "\"INNERTUBE_CLIENT_VERSION\":\"", "\"").map(str::to_string)
}

pub fn initial_data(html: &str) -> Option<Value> {
    serde_json::from_str(between(html, "var ytInitialData = ", ";</script>")?).ok()
}

pub fn selected_tab(data: &Value) -> Option<&Value> {
    data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"]
        .as_array()?
        .iter()
        .find(|tab| tab["tabRenderer"]["selected"] == true)
}

pub fn alert_text(data: &Value) -> String {
    let mut alerts = Vec::new();
    walk(&data["alerts"], "simpleText", &mut alerts);
    alerts
        .into_iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
        .join(" ")
}

fn walk<'a>(value: &'a Value, key: &str, found: &mut Vec<&'a Value>) {
    match value {
        Value::Object(map) => {
            for (name, child) in map {
                if name == key {
                    found.push(child);
                } else {
                    walk(child, key, found);
                }
            }
        }
        Value::Array(items) => {
            for child in items {
                walk(child, key, found);
            }
        }
        _ => {}
    }
}

pub fn videos(value: &Value) -> Vec<Video> {
    let mut lockups = Vec::new();
    walk(value, "lockupViewModel", &mut lockups);
    lockups.into_iter().filter_map(lockup_video).collect()
}

pub fn continuation_token(value: &Value) -> Option<String> {
    let mut items = Vec::new();
    walk(value, "continuationItemRenderer", &mut items);
    items.into_iter().find_map(|item| {
        item["continuationEndpoint"]["continuationCommand"]["token"]
            .as_str()
            .map(str::to_string)
    })
}

fn lockup_video(lockup: &Value) -> Option<Video> {
    let id = lockup["contentId"].as_str()?;
    let metadata = &lockup["metadata"]["lockupMetadataViewModel"];
    let title = metadata["title"]["content"].as_str()?;
    let parts: Vec<&str> = metadata["metadata"]["contentMetadataViewModel"]["metadataRows"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|row| row["metadataParts"].as_array().into_iter().flatten())
        .filter_map(|part| part["text"]["content"].as_str())
        .collect();
    let mut badges = Vec::new();
    walk(&lockup["contentImage"], "thumbnailBadgeViewModel", &mut badges);
    let duration_text = badges
        .into_iter()
        .filter_map(|badge| badge["text"].as_str())
        .find(|text| text.contains(':'))
        .unwrap_or("");
    let age_text = parts.iter().find(|part| part.ends_with("ago")).unwrap_or(&"");
    let views_text = parts.iter().find(|part| !part.ends_with("ago")).unwrap_or(&"");
    Some(Video {
        id: id.to_string(),
        title: title.to_string(),
        description: String::new(),
        keywords: Vec::new(),
        duration_text: duration_text.to_string(),
        duration_seconds: parse_duration(duration_text),
        views: parse_views(views_text),
        age_text: age_text.to_string(),
    })
}

pub fn parse_duration(text: &str) -> u64 {
    text.split(':')
        .try_fold(0, |total, part| {
            part.trim().parse::<u64>().ok().map(|n| total * 60 + n)
        })
        .unwrap_or(0)
}

pub fn parse_views(text: &str) -> u64 {
    let token = text.split_whitespace().next().unwrap_or("").replace(',', "");
    let (number, scale) = match token.chars().last() {
        Some('K') => (&token[..token.len() - 1], 1_000.0),
        Some('M') => (&token[..token.len() - 1], 1_000_000.0),
        Some('B') => (&token[..token.len() - 1], 1_000_000_000.0),
        _ => (token.as_str(), 1.0),
    };
    number
        .parse::<f64>()
        .map(|n| (n * scale).round() as u64)
        .unwrap_or(0)
}

pub fn details(player: &Value) -> Details {
    let video = &player["videoDetails"];
    let number = |key: &str| video[key].as_str().and_then(|text| text.parse::<u64>().ok());
    Details {
        description: video["shortDescription"].as_str().unwrap_or("").to_string(),
        keywords: video["keywords"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|keyword| keyword.as_str().map(str::to_string))
            .collect(),
        duration_seconds: number("lengthSeconds"),
        views: number("viewCount"),
    }
}

pub fn apply_details(video: &mut Video, details: Details) {
    video.description = details.description;
    video.keywords = details.keywords;
    if let Some(seconds) = details.duration_seconds {
        video.duration_seconds = seconds;
    }
    if let Some(views) = details.views {
        video.views = views;
    }
}

async fn post(http: &reqwest::Client, url: &str, body: Value) -> Result<Value> {
    Ok(http
        .post(url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

fn browser_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT_LANGUAGE, HeaderValue::from_static("en-US,en;q=0.9"));
    headers.insert(USER_AGENT, HeaderValue::from_static(BROWSER));
    headers.insert(COOKIE, HeaderValue::from_static("CONSENT=YES+1; SOCS=CAI"));
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

pub async fn fetch_videos() -> Result<Vec<Video>> {
    let http = browser_client()?;
    let html = http
        .get(CHANNEL_URL)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let version = client_version(&html).context("youtube client version not found")?;
    let data = initial_data(&html).context("youtube initial data not found")?;
    let tab = selected_tab(&data).with_context(|| {
        format!("youtube videos tab not found: {}", alert_text(&data))
    })?;
    let context = json!({"client": {"clientName": "WEB", "clientVersion": version, "hl": "en", "gl": "US"}});
    let mut all = videos(tab);
    let mut token = continuation_token(tab);
    while let Some(continuation) = token {
        let body = json!({"context": context, "continuation": continuation});
        let page = post(&http, BROWSE_URL, body).await?;
        let found = videos(&page);
        if found.is_empty() {
            break;
        }
        all.extend(found);
        token = continuation_token(&page);
    }
    for start in (0..all.len()).step_by(CONCURRENCY) {
        let mut tasks = JoinSet::new();
        for index in start..all.len().min(start + CONCURRENCY) {
            let http = http.clone();
            let body = json!({"context": context, "videoId": all[index].id});
            tasks.spawn(async move { (index, post(&http, PLAYER_URL, body).await) });
        }
        while let Some(done) = tasks.join_next().await {
            let (index, player) = done?;
            if let Ok(player) = player {
                apply_details(&mut all[index], details(&player));
            }
        }
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lockup(id: &str, title: &str, duration: &str, views: &str, age: &str) -> Value {
        json!({"lockupViewModel": {
            "contentId": id,
            "contentImage": {"thumbnailViewModel": {"overlays": [
                {"thumbnailBottomOverlayViewModel": {"badges": [
                    {"thumbnailBadgeViewModel": {"text": duration}}
                ]}}
            ]}},
            "metadata": {"lockupMetadataViewModel": {
                "title": {"content": title},
                "metadata": {"contentMetadataViewModel": {"metadataRows": [
                    {"metadataParts": [{"text": {"content": views}}, {"text": {"content": age}}]}
                ]}}
            }}
        }})
    }

    fn page() -> Value {
        json!({"contents": {"twoColumnBrowseResultsRenderer": {"tabs": [
            {"tabRenderer": {"title": "Home", "content": {"items": [
                lockup("home0000001", "Not in the videos tab", "1:00", "5", "1y ago")
            ]}}},
            {"tabRenderer": {"selected": true, "content": {"richGridRenderer": {
                "header": {"chips": [{"chipViewModel": {"tapCommand": {
                    "continuationCommand": {"token": "CHIP_TOKEN"}
                }}}]},
                "contents": [
                    {"richItemRenderer": {"content": lockup("aaaaaaaaaaa", "Rust ownership", "7:02", "104", "10mo ago")}},
                    {"richItemRenderer": {"content": lockup("bbbbbbbbbbb", "Zig comptime", "1:02:03", "1.2K views", "2 years ago")}},
                    {"continuationItemRenderer": {"continuationEndpoint": {
                        "continuationCommand": {"token": "NEXT_PAGE_TOKEN"}
                    }}}
                ]
            }}}}
        ]}}})
    }

    #[test]
    fn only_the_selected_tab_feeds_the_index() {
        let data = page();
        let found = videos(selected_tab(&data).unwrap());
        let ids: Vec<&str> = found.iter().map(|video| video.id.as_str()).collect();
        assert_eq!(ids, ["aaaaaaaaaaa", "bbbbbbbbbbb"]);
    }

    #[test]
    fn lockup_fields_become_the_video_shown_in_results() {
        let data = page();
        let found = videos(selected_tab(&data).unwrap());
        assert_eq!(found[0].title, "Rust ownership");
        assert_eq!(found[0].duration_text, "7:02");
        assert_eq!(found[0].duration_seconds, 422);
        assert_eq!(found[0].views, 104);
        assert_eq!(found[0].age_text, "10mo ago");
        assert_eq!(found[1].duration_seconds, 3723);
        assert_eq!(found[1].views, 1200);
        assert_eq!(found[1].age_text, "2 years ago");
    }

    #[test]
    fn paging_follows_the_grid_continuation_and_never_a_filter_chip() {
        let data = page();
        let token = continuation_token(selected_tab(&data).unwrap());
        assert_eq!(token.as_deref(), Some("NEXT_PAGE_TOKEN"));
    }

    #[test]
    fn last_page_has_no_token_so_paging_stops() {
        let last = json!({"items": [lockup("ccccccccccc", "Last", "0:30", "1", "1d ago")]});
        assert_eq!(continuation_token(&last), None);
        assert_eq!(videos(&last).len(), 1);
    }

    #[test]
    fn duration_text_becomes_seconds_for_the_range_filter() {
        assert_eq!(parse_duration("7:02"), 422);
        assert_eq!(parse_duration("0:45"), 45);
        assert_eq!(parse_duration("1:02:03"), 3723);
        assert_eq!(parse_duration(""), 0);
        assert_eq!(parse_duration("LIVE"), 0);
    }

    #[test]
    fn views_text_becomes_an_integer_payload() {
        assert_eq!(parse_views("104"), 104);
        assert_eq!(parse_views("104 views"), 104);
        assert_eq!(parse_views("1,234 views"), 1234);
        assert_eq!(parse_views("1.2K views"), 1200);
        assert_eq!(parse_views("3M views"), 3_000_000);
        assert_eq!(parse_views("No views"), 0);
    }

    #[test]
    fn page_scraping_finds_version_and_initial_data() {
        let html = r#"<script>ytcfg.set({"INNERTUBE_CLIENT_VERSION":"2.20261001.00.00","X":1});</script><script>var ytInitialData = {"contents":{"k":"a;b"}};</script><script>other</script>"#;
        assert_eq!(client_version(html).as_deref(), Some("2.20261001.00.00"));
        assert_eq!(initial_data(html).unwrap()["contents"]["k"], "a;b");
        assert_eq!(client_version("<html></html>"), None);
        assert_eq!(initial_data("<html></html>"), None);
    }

    #[test]
    fn a_rejected_channel_reports_the_reason_youtube_gave() {
        let data = json!({"alerts": [{"alertRenderer": {
            "type": "ERROR",
            "text": {"simpleText": "This channel does not exist."}
        }}]});
        assert_eq!(selected_tab(&data), None);
        assert_eq!(alert_text(&data), "This channel does not exist.");
        assert_eq!(alert_text(&page()), "");
    }

    #[test]
    fn player_details_survive_an_unplayable_status() {
        let player = json!({
            "playabilityStatus": {"status": "UNPLAYABLE"},
            "videoDetails": {
                "shortDescription": "Deep dive into borrowing",
                "lengthSeconds": "425",
                "viewCount": "110",
                "keywords": ["rust", "borrow checker"]
            }
        });
        assert_eq!(
            details(&player),
            Details {
                description: "Deep dive into borrowing".to_string(),
                keywords: vec!["rust".to_string(), "borrow checker".to_string()],
                duration_seconds: Some(425),
                views: Some(110),
            }
        );
    }

    #[test]
    fn missing_player_details_keep_the_listing_values() {
        let data = page();
        let mut video = videos(selected_tab(&data).unwrap()).remove(0);
        apply_details(&mut video, details(&json!({"playabilityStatus": {"status": "ERROR"}})));
        assert_eq!(video.description, "");
        assert_eq!(video.duration_seconds, 422);
        assert_eq!(video.views, 104);
    }
}

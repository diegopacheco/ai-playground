use anyhow::Result;
use qdrant_client::qdrant::{
    Condition, CountPointsBuilder, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder,
    Distance, FieldType, Filter, PointStruct, Query, QueryPointsBuilder, Range,
    RecommendInputBuilder, ScoredPoint, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use serde::Serialize;
use serde_json::json;

use crate::youtube::Video;

pub const COLLECTION: &str = "videos";
const VECTOR_SIZE: u64 = 768;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Hit {
    pub id: String,
    pub score: f32,
    pub video_id: String,
    pub title: String,
    pub description: String,
    pub url: String,
    pub thumbnail: String,
    pub duration_text: String,
    pub views: i64,
    pub age_text: String,
}

pub struct Store {
    client: Qdrant,
}

fn fnv1a(seed: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(seed, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

pub fn point_id(video_id: &str) -> String {
    let high = fnv1a(0xcbf2_9ce4_8422_2325, video_id.as_bytes());
    let low = fnv1a(high, video_id.as_bytes());
    let hex = format!("{high:016x}{low:016x}");
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

pub fn duration_filter(min_minutes: Option<f64>, max_minutes: Option<f64>) -> Option<Filter> {
    if min_minutes.is_none() && max_minutes.is_none() {
        return None;
    }
    let range = Range {
        gte: min_minutes.map(|minutes| minutes * 60.0),
        lte: max_minutes.map(|minutes| minutes * 60.0),
        ..Default::default()
    };
    Some(Filter::must([Condition::range("duration_seconds", range)]))
}

fn point(video: &Video, vector: Vec<f32>) -> Result<PointStruct> {
    let payload = Payload::try_from(json!({
        "video_id": video.id,
        "title": video.title,
        "description": video.description,
        "url": format!("https://www.youtube.com/watch?v={}", video.id),
        "thumbnail": format!("https://i.ytimg.com/vi/{}/hqdefault.jpg", video.id),
        "duration_seconds": video.duration_seconds,
        "duration_text": video.duration_text,
        "views": video.views,
        "age_text": video.age_text,
    }))?;
    Ok(PointStruct::new(point_id(&video.id), vector, payload))
}

fn hit(point: ScoredPoint) -> Hit {
    let text = |key: &str| {
        point
            .payload
            .get(key)
            .and_then(|value| value.as_str())
            .map(|value| value.to_string())
            .unwrap_or_default()
    };
    let video_id = text("video_id");
    Hit {
        id: point_id(&video_id),
        score: point.score,
        video_id,
        title: text("title"),
        description: text("description"),
        url: text("url"),
        thumbnail: text("thumbnail"),
        duration_text: text("duration_text"),
        views: point
            .payload
            .get("views")
            .and_then(|value| value.as_integer())
            .unwrap_or(0),
        age_text: text("age_text"),
    }
}

impl Store {
    pub fn connect(url: &str) -> Result<Self> {
        Ok(Self {
            client: Qdrant::from_url(url).build()?,
        })
    }

    pub async fn ensure_collection(&self) -> Result<()> {
        if self.client.collection_exists(COLLECTION).await? {
            return Ok(());
        }
        self.client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION)
                    .vectors_config(VectorParamsBuilder::new(VECTOR_SIZE, Distance::Cosine)),
            )
            .await?;
        self.client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(
                    COLLECTION,
                    "duration_seconds",
                    FieldType::Integer,
                )
                .wait(true),
            )
            .await?;
        Ok(())
    }

    pub async fn upsert(&self, videos: &[Video], vectors: Vec<Vec<f32>>) -> Result<()> {
        let points = videos
            .iter()
            .zip(vectors)
            .map(|(video, vector)| point(video, vector))
            .collect::<Result<Vec<_>>>()?;
        self.client
            .upsert_points(UpsertPointsBuilder::new(COLLECTION, points).wait(true))
            .await?;
        Ok(())
    }

    pub async fn count(&self) -> Result<u64> {
        let response = self
            .client
            .count(CountPointsBuilder::new(COLLECTION).exact(true))
            .await?;
        Ok(response.result.map(|result| result.count).unwrap_or(0))
    }

    async fn query(&self, request: QueryPointsBuilder) -> Result<Vec<Hit>> {
        let response = self.client.query(request.with_payload(true)).await?;
        Ok(response.result.into_iter().map(hit).collect())
    }

    pub async fn search(
        &self,
        vector: Vec<f32>,
        limit: u64,
        min_minutes: Option<f64>,
        max_minutes: Option<f64>,
    ) -> Result<Vec<Hit>> {
        let mut request = QueryPointsBuilder::new(COLLECTION)
            .query(Query::new_nearest(vector))
            .limit(limit);
        if let Some(filter) = duration_filter(min_minutes, max_minutes) {
            request = request.filter(filter);
        }
        self.query(request).await
    }

    pub async fn similar(&self, video_id: &str, limit: u64) -> Result<Vec<Hit>> {
        let recommend = RecommendInputBuilder::default()
            .add_positive(point_id(video_id))
            .build();
        self.query(
            QueryPointsBuilder::new(COLLECTION)
                .query(Query::new_recommend(recommend))
                .limit(limit),
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_id_is_stable_so_reingest_overwrites_instead_of_duplicating() {
        assert_eq!(point_id("dQw4w9WgXcQ"), point_id("dQw4w9WgXcQ"));
        assert_eq!(point_id("dQw4w9WgXcQ"), "a09eed42-6133-21dd-695c-30bc82d5fa25");
    }

    #[test]
    fn point_id_differs_per_video_so_videos_never_overwrite_each_other() {
        assert_ne!(point_id("aaaaaaaaaaa"), point_id("aaaaaaaaaab"));
    }

    #[test]
    fn point_id_has_the_uuid_shape_qdrant_requires() {
        let id = point_id("aaaaaaaaaaa");
        let groups: Vec<usize> = id.split('-').map(str::len).collect();
        assert_eq!(groups, [8, 4, 4, 4, 12]);
        assert!(id.chars().all(|c| c == '-' || c.is_ascii_hexdigit()));
    }

    #[test]
    fn no_minutes_means_no_filter_so_every_video_is_searchable() {
        assert_eq!(duration_filter(None, None), None);
    }

    #[test]
    fn minutes_become_a_seconds_range_on_the_indexed_field() {
        let expected = Filter::must([Condition::range(
            "duration_seconds",
            Range {
                gte: Some(300.0),
                lte: Some(900.0),
                ..Default::default()
            },
        )]);
        assert_eq!(duration_filter(Some(5.0), Some(15.0)), Some(expected));
        let open = Filter::must([Condition::range(
            "duration_seconds",
            Range {
                lte: Some(120.0),
                ..Default::default()
            },
        )]);
        assert_eq!(duration_filter(None, Some(2.0)), Some(open));
    }
}

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;
use std::sync::Arc;
use k8s_openapi::api::core::v1::{Pod, Service};
use k8s_openapi::api::apps::v1::{Deployment, ReplicaSet};
use kube::api::{Api, ListParams};
use crate::AppState;

#[derive(Serialize)]
pub struct ClusterObject {
    pub namespace: String,
    pub kind: String,
    pub name: String,
    pub status: String,
    pub ready: String,
    pub restarts: i32,
    pub age: String,
    pub yaml: String,
}

pub async fn get_api_status(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<ClusterObject>>, (StatusCode, String)> {
    let client = &state.k8s_client;
    let mut objects: Vec<ClusterObject> = Vec::new();

    let pods: Api<Pod> = Api::all(client.clone());
    let pod_list = pods.list(&ListParams::default()).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    for pod in pod_list {
        let name = pod.metadata.name.clone().unwrap_or_default();
        let ns = pod.metadata.namespace.clone().unwrap_or_default();

        let phase = pod.status.as_ref()
            .and_then(|s| s.phase.clone())
            .unwrap_or("Unknown".to_string());

        let mut status_str = phase.clone();
        let mut ready_count = 0;
        let mut total_count = 0;
        let mut restarts = 0;

        if let Some(st) = &pod.status {
            if let Some(css) = &st.container_statuses {
                total_count = css.len();
                for cs in css {
                    if cs.ready { ready_count += 1; }
                    restarts += cs.restart_count;
                    if let Some(state) = &cs.state {
                        if let Some(w) = &state.waiting {
                            status_str = w.reason.clone().unwrap_or(status_str);
                        }
                    }
                }
            }
        }

        let age = pod.metadata.creation_timestamp.as_ref()
            .map(|t| format_age(&t.0))
            .unwrap_or_default();

        let yaml = serde_yaml::to_string(&pod).unwrap_or_default();

        objects.push(ClusterObject {
            namespace: ns,
            kind: "Pod".to_string(),
            name,
            status: status_str,
            ready: format!("{}/{}", ready_count, total_count),
            restarts,
            age,
            yaml,
        });
    }

    let deployments: Api<Deployment> = Api::all(client.clone());
    let dep_list = deployments.list(&ListParams::default()).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    for dep in dep_list {
        let name = dep.metadata.name.clone().unwrap_or_default();
        let ns = dep.metadata.namespace.clone().unwrap_or_default();
        let ready = dep.status.as_ref().and_then(|s| s.ready_replicas).unwrap_or(0);
        let desired = dep.spec.as_ref().and_then(|s| s.replicas).unwrap_or(1);
        let available = dep.status.as_ref().and_then(|s| s.available_replicas).unwrap_or(0);
        let status_str = if available >= desired { "Available".to_string() } else { "Progressing".to_string() };
        let age = dep.metadata.creation_timestamp.as_ref()
            .map(|t| format_age(&t.0))
            .unwrap_or_default();
        let yaml = serde_yaml::to_string(&dep).unwrap_or_default();

        objects.push(ClusterObject {
            namespace: ns,
            kind: "Deployment".to_string(),
            name,
            status: status_str,
            ready: format!("{}/{}", ready, desired),
            restarts: 0,
            age,
            yaml,
        });
    }

    let services: Api<Service> = Api::all(client.clone());
    let svc_list = services.list(&ListParams::default()).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    for svc in svc_list {
        let name = svc.metadata.name.clone().unwrap_or_default();
        let ns = svc.metadata.namespace.clone().unwrap_or_default();
        let svc_type = svc.spec.as_ref()
            .and_then(|s| s.type_.clone())
            .unwrap_or("ClusterIP".to_string());
        let age = svc.metadata.creation_timestamp.as_ref()
            .map(|t| format_age(&t.0))
            .unwrap_or_default();
        let yaml = serde_yaml::to_string(&svc).unwrap_or_default();

        objects.push(ClusterObject {
            namespace: ns,
            kind: "Service".to_string(),
            name,
            status: svc_type,
            ready: "-".to_string(),
            restarts: 0,
            age,
            yaml,
        });
    }

    let replicasets: Api<ReplicaSet> = Api::all(client.clone());
    let rs_list = replicasets.list(&ListParams::default()).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    for rs in rs_list {
        let name = rs.metadata.name.clone().unwrap_or_default();
        let ns = rs.metadata.namespace.clone().unwrap_or_default();
        let ready = rs.status.as_ref().and_then(|s| s.ready_replicas).unwrap_or(0);
        let desired = rs.spec.as_ref().and_then(|s| s.replicas).unwrap_or(0);
        let age = rs.metadata.creation_timestamp.as_ref()
            .map(|t| format_age(&t.0))
            .unwrap_or_default();
        let yaml = serde_yaml::to_string(&rs).unwrap_or_default();

        objects.push(ClusterObject {
            namespace: ns,
            kind: "ReplicaSet".to_string(),
            name,
            status: if ready >= desired { "Ready".to_string() } else { "NotReady".to_string() },
            ready: format!("{}/{}", ready, desired),
            restarts: 0,
            age,
            yaml,
        });
    }

    Ok(Json(objects))
}

fn format_age(created: &chrono::DateTime<chrono::Utc>) -> String {
    let dur = chrono::Utc::now().signed_duration_since(created);
    if dur.num_days() > 0 {
        format!("{}d", dur.num_days())
    } else if dur.num_hours() > 0 {
        format!("{}h", dur.num_hours())
    } else if dur.num_minutes() > 0 {
        format!("{}m", dur.num_minutes())
    } else {
        format!("{}s", dur.num_seconds())
    }
}

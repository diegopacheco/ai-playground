use k8s_openapi::api::apps::v1::Deployment;
use k8s_openapi::api::core::v1::Pod;
use kube::api::{Api, ListParams, LogParams};
use kube::Client;

pub async fn collect_all_logs(client: &Client) -> Result<String, kube::Error> {
    let pods: Api<Pod> = Api::all(client.clone());
    let pod_list = pods.list(&ListParams::default()).await?;
    let mut output = String::new();

    for pod in pod_list {
        let name = pod.metadata.name.clone().unwrap_or_default();
        let ns = pod.metadata.namespace.clone().unwrap_or("default".to_string());
        let pod_api: Api<Pod> = Api::namespaced(client.clone(), &ns);

        let log_params = LogParams {
            tail_lines: Some(50),
            ..Default::default()
        };

        match pod_api.logs(&name, &log_params).await {
            Ok(logs) => {
                if !logs.trim().is_empty() {
                    output.push_str(&format!("=== Pod: {}/{} ===\n{}\n\n", ns, name, logs));
                }
            }
            Err(e) => {
                output.push_str(&format!("=== Pod: {}/{} === ERROR: {}\n\n", ns, name, e));
            }
        }
    }

    if output.is_empty() {
        output = "No logs found.".to_string();
    }

    Ok(output)
}

pub async fn collect_diagnostics(client: &Client) -> Result<String, kube::Error> {
    let mut diag = String::new();

    let pods: Api<Pod> = Api::namespaced(client.clone(), "default");
    let pod_list = pods.list(&ListParams::default()).await?;

    for pod in &pod_list {
        let name = pod.metadata.name.clone().unwrap_or_default();

        if name.starts_with("sre-agent") {
            continue;
        }

        let phase = pod.status.as_ref()
            .and_then(|s| s.phase.clone())
            .unwrap_or("Unknown".to_string());

        let mut unhealthy = false;

        if let Some(status) = &pod.status {
            if let Some(container_statuses) = &status.container_statuses {
                for cs in container_statuses {
                    if cs.ready == false || cs.restart_count > 0 {
                        unhealthy = true;
                    }
                    if let Some(state) = &cs.state {
                        if state.waiting.is_some() || state.terminated.is_some() {
                            unhealthy = true;
                        }
                    }
                }
            }
        }

        if phase != "Running" || unhealthy {
            diag.push_str(&format!("--- Pod: {} (phase: {}) ---\n", name, phase));

            if let Some(status) = &pod.status {
                if let Some(conditions) = &status.conditions {
                    for c in conditions {
                        diag.push_str(&format!("  Condition: {} = {} ({})\n",
                            c.type_, c.status,
                            c.message.as_deref().unwrap_or("")));
                    }
                }
                if let Some(container_statuses) = &status.container_statuses {
                    for cs in container_statuses {
                        diag.push_str(&format!("  Container: {} ready={} restarts={}\n",
                            cs.name, cs.ready, cs.restart_count));
                        if let Some(state) = &cs.state {
                            if let Some(w) = &state.waiting {
                                diag.push_str(&format!("    Waiting: {} ({})\n",
                                    w.reason.as_deref().unwrap_or(""),
                                    w.message.as_deref().unwrap_or("")));
                            }
                            if let Some(t) = &state.terminated {
                                diag.push_str(&format!("    Terminated: reason={} exit_code={}\n",
                                    t.reason.as_deref().unwrap_or(""),
                                    t.exit_code));
                            }
                        }
                    }
                }
            }

            let log_params = LogParams {
                tail_lines: Some(20),
                ..Default::default()
            };
            match pods.logs(&name, &log_params).await {
                Ok(logs) if !logs.trim().is_empty() => {
                    diag.push_str(&format!("  Logs:\n{}\n", logs));
                }
                _ => {}
            }

            diag.push('\n');
        }
    }

    let deployments: Api<Deployment> = Api::namespaced(client.clone(), "default");
    let dep_list = deployments.list(&ListParams::default()).await?;

    for dep in &dep_list {
        let name = dep.metadata.name.clone().unwrap_or_default();
        if name.starts_with("sre-agent") {
            continue;
        }

        let ready = dep.status.as_ref()
            .and_then(|s| s.ready_replicas)
            .unwrap_or(0);
        let desired = dep.spec.as_ref()
            .and_then(|s| s.replicas)
            .unwrap_or(1);

        if ready < desired {
            diag.push_str(&format!("--- Deployment: {} (ready: {}/{}) ---\n", name, ready, desired));

            if let Some(status) = &dep.status {
                if let Some(conditions) = &status.conditions {
                    for c in conditions {
                        diag.push_str(&format!("  Condition: {} = {} ({})\n",
                            c.type_, c.status,
                            c.message.as_deref().unwrap_or("")));
                    }
                }
            }

            if let Some(spec) = &dep.spec {
                if let Some(template) = &spec.template.spec {
                    for c in &template.containers {
                        diag.push_str(&format!("  Container spec: name={} image={}\n",
                            c.name, c.image.as_deref().unwrap_or("none")));
                        if let Some(ports) = &c.ports {
                            for p in ports {
                                diag.push_str(&format!("    port: {}\n", p.container_port));
                            }
                        }
                        if let Some(probe) = &c.readiness_probe {
                            if let Some(http) = &probe.http_get {
                                diag.push_str(&format!("    readinessProbe: path={} port={:?}\n",
                                    http.path.as_deref().unwrap_or(""),
                                    http.port));
                            }
                        }
                        if let Some(probe) = &c.liveness_probe {
                            if let Some(http) = &probe.http_get {
                                diag.push_str(&format!("    livenessProbe: path={} port={:?}\n",
                                    http.path.as_deref().unwrap_or(""),
                                    http.port));
                            }
                        }
                    }
                }
            }

            diag.push('\n');
        }
    }

    Ok(diag)
}

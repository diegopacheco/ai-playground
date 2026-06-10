use crate::json;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct Agent {
    pub pid: i64,
    pub status: String,
    pub project: String,
    pub model: String,
    pub branch: String,
    pub version: String,
    pub started_at: i64,
    pub cpu: f64,
    pub mem: f64,
    pub ctx_tokens: i64,
    pub out_tokens: i64,
    pub msgs: i64,
}

struct Stat {
    len: u64,
    ctx: i64,
    out: i64,
    msgs: i64,
    model: String,
    branch: String,
}

pub struct Cache {
    map: HashMap<String, Stat>,
}

impl Cache {
    pub fn new() -> Cache {
        Cache { map: HashMap::new() }
    }
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/".into()))
}

pub fn collect(cache: &mut Cache) -> Vec<Agent> {
    let base = home().join(".claude");
    let sessions_dir = base.join("sessions");
    let projects_dir = base.join("projects");

    let mut raw: Vec<(i64, String, String, String, i64, String)> = Vec::new();
    if let Ok(entries) = fs::read_dir(&sessions_dir) {
        for e in entries.flatten() {
            let path = e.path();
            if path.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let Ok(txt) = fs::read_to_string(&path) else { continue };
            let Some(pid) = json::get_i64(&txt, "pid") else { continue };
            let session_id = json::get_str(&txt, "sessionId").unwrap_or_default();
            let cwd = json::get_str(&txt, "cwd").unwrap_or_default();
            let status = json::get_str(&txt, "status").unwrap_or_else(|| "?".into());
            let started_at = json::get_i64(&txt, "startedAt").unwrap_or(0);
            let version = json::get_str(&txt, "version").unwrap_or_default();
            raw.push((pid, session_id, cwd, status, started_at, version));
        }
    }

    let pids: Vec<i64> = raw.iter().map(|r| r.0).collect();
    let res = resources(&pids);

    let mut agents = Vec::new();
    for (pid, session_id, cwd, status, started_at, version) in raw {
        let Some(&(cpu, mem, _rss)) = res.get(&pid) else { continue };
        let project = cwd.rsplit('/').next().unwrap_or(&cwd).to_string();
        let stat = transcript_stats(&projects_dir, &cwd, &session_id, cache);
        agents.push(Agent {
            pid,
            status,
            project,
            model: stat.0,
            branch: stat.1,
            version,
            started_at,
            cpu,
            mem,
            ctx_tokens: stat.2,
            out_tokens: stat.3,
            msgs: stat.4,
        });
    }
    agents
}

fn resources(pids: &[i64]) -> HashMap<i64, (f64, f64, i64)> {
    let mut out = HashMap::new();
    if pids.is_empty() {
        return out;
    }
    let csv = pids
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let Ok(o) = Command::new("ps")
        .args(["-o", "pid=,%cpu=,%mem=,rss=", "-p", &csv])
        .output()
    else {
        return out;
    };
    let text = String::from_utf8_lossy(&o.stdout);
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 4 {
            continue;
        }
        let pid = f[0].parse::<i64>().unwrap_or(-1);
        let cpu = f[1].parse::<f64>().unwrap_or(0.0);
        let mem = f[2].parse::<f64>().unwrap_or(0.0);
        let rss = f[3].parse::<i64>().unwrap_or(0);
        out.insert(pid, (cpu, mem, rss));
    }
    out
}

fn encode_path(cwd: &str) -> String {
    cwd.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

fn transcript_stats(
    projects_dir: &PathBuf,
    cwd: &str,
    session_id: &str,
    cache: &mut Cache,
) -> (String, String, i64, i64, i64) {
    let path = projects_dir
        .join(encode_path(cwd))
        .join(format!("{}.jsonl", session_id));
    let len = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    if let Some(s) = cache.map.get(session_id) {
        if s.len == len {
            return (s.model.clone(), s.branch.clone(), s.ctx, s.out, s.msgs);
        }
    }
    let Ok(content) = fs::read_to_string(&path) else {
        return (String::new(), String::new(), 0, 0, 0);
    };
    let mut ctx = 0;
    let mut out = 0;
    let mut msgs = 0;
    let mut model = String::new();
    let mut branch = String::new();
    for line in content.lines() {
        if let Some(b) = json::get_str(line, "gitBranch") {
            if !b.is_empty() {
                branch = b;
            }
        }
        let assistant = line.contains("\"role\":\"assistant\"");
        if assistant || line.contains("\"role\":\"user\"") {
            msgs += 1;
        }
        if assistant {
            if let Some(m) = json::get_str(line, "model") {
                if !m.is_empty() && m != "<synthetic>" {
                    model = m;
                }
            }
            if let Some(i) = json::get_i64(line, "input_tokens") {
                let cr = json::get_i64(line, "cache_read_input_tokens").unwrap_or(0);
                let cc = json::get_i64(line, "cache_creation_input_tokens").unwrap_or(0);
                ctx = i + cr + cc;
            }
            out += json::get_i64(line, "output_tokens").unwrap_or(0);
        }
    }
    cache.map.insert(
        session_id.to_string(),
        Stat {
            len,
            ctx,
            out,
            msgs,
            model: model.clone(),
            branch: branch.clone(),
        },
    );
    (model, branch, ctx, out, msgs)
}

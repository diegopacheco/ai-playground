use crate::persistence::models::{DriftRecord, DriftReport};
use crate::drift::embeddings::{build_shared_vocab, shared_vocab_embedding, cosine_similarity};

const DRIFT_THRESHOLD: f64 = 0.75;

pub fn analyze_drift(records: &[DriftRecord]) -> Vec<DriftReport> {
    if records.len() < 2 {
        return vec![];
    }

    let texts: Vec<&str> = records.iter().map(|r| r.response.as_str()).collect();
    let vocab = build_shared_vocab(&texts);

    let embeddings: Vec<Vec<f64>> = texts.iter()
        .map(|t| shared_vocab_embedding(t, &vocab))
        .collect();

    let baseline = &embeddings[0];

    let mut reports = Vec::new();
    for i in 1..records.len() {
        let sim = cosine_similarity(baseline, &embeddings[i]);
        reports.push(DriftReport {
            date: records[i].created_at.clone(),
            cosine_similarity: sim,
            drift_detected: sim < DRIFT_THRESHOLD,
        });
    }
    reports
}

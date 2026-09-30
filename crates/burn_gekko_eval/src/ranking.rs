//! Tie-aware AUROC and average precision for co-visibility scores.
use anyhow::{Result, ensure};
use serde::Serialize;
#[derive(Clone, Debug, Serialize)]
pub struct RankingMetrics {
    pub pixels: usize,
    pub positives: usize,
    pub average_precision: Option<f64>,
    pub auroc: Option<f64>,
}

/// Group equal scores before advancing thresholds, so arbitrary tie ordering cannot inflate AP/AUC.
pub fn ranking_metrics(values: &[(f32, bool)]) -> Result<RankingMetrics> {
    ensure!(
        values.iter().all(|(s, _)| s.is_finite()),
        "nonfinite RI score"
    );
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.0.total_cmp(&a.0));
    let positives = sorted.iter().filter(|(_, p)| *p).count();
    let negatives = sorted.len() - positives;
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut ap = 0.0;
    let mut auc = 0.0;
    let mut i = 0;
    while i < sorted.len() {
        let mut end = i + 1;
        while end < sorted.len() && sorted[end].0 == sorted[i].0 {
            end += 1;
        }
        let p = sorted[i..end].iter().filter(|(_, p)| *p).count();
        let n = end - i - p;
        tp += p;
        fp += n;
        if positives > 0 {
            ap += p as f64 / positives as f64 * tp as f64 / (tp + fp) as f64;
        }
        auc += p as f64 * ((negatives - fp) as f64 + 0.5 * n as f64);
        i = end;
    }
    Ok(RankingMetrics {
        pixels: sorted.len(),
        positives,
        average_precision: (positives > 0).then_some(ap),
        auroc: (positives > 0 && negatives > 0).then(|| auc / (positives * negatives) as f64),
    })
}

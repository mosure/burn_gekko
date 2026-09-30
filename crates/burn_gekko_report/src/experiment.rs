use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedFile {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentEvidence {
    pub directory: PathBuf,
    pub metrics_sha256: String,
    pub provenance_sha256: String,
    pub evaluation_use: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EfficiencyEvidence {
    pub ledger: PinnedFile,
    pub telemetry: PinnedFile,
    pub command: String,
}
/// Intentionally singular: private candidate-vs-candidate publishing is not supported.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Experiment {
    pub schema: u32,
    pub id: String,
    pub title: String,
    pub author: String,
    pub description: String,
    pub architecture: String,
    pub run: PathBuf,
    pub checkpoint: PathBuf,
    pub checkpoint_sha256: String,
    pub latent: LatentEvidence,
    #[serde(default)]
    pub benchmarks: Vec<PinnedFile>,
    /// Future heads use the same schema and must bind to this checkpoint.
    #[serde(default)]
    pub heads: Vec<PinnedFile>,
    /// Optional RGB-transform diagnostic with its own geometric visualizations.
    #[serde(default)]
    pub equivariance: Option<PinnedFile>,
    pub efficiency: Option<EfficiencyEvidence>,
    pub limitations: Vec<String>,
    /// Deterministic first/middle/last exported samples, never chosen by quality.
    pub sample_count: usize,
}

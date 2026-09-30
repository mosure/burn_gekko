//! Extensible capability records shared by scoring and publication.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityStatus {
    Evaluated,
    NotEvaluated,
    NotTrained,
    NotImplemented,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metric {
    pub id: String,
    pub label: String,
    pub value: f64,
    pub unit: String,
    pub lower_is_better: bool,
    pub samples: usize,
    pub aggregation: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capability {
    /// Stable task/head identifier; adding a decoder does not change the report schema.
    pub id: String,
    pub label: String,
    pub status: CapabilityStatus,
    pub protocol: String,
    pub limitations: Vec<String>,
    #[serde(default)]
    pub metrics: Vec<Metric>,
}
impl Capability {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.id.is_empty() && !self.protocol.is_empty(),
            "capability requires id and protocol"
        );
        ensure!(
            (self.status == CapabilityStatus::Evaluated) == !self.metrics.is_empty(),
            "only evaluated capabilities may contain metrics, and evaluated capabilities require metrics"
        );
        for m in &self.metrics {
            ensure!(
                m.value.is_finite()
                    && m.samples > 0
                    && !m.unit.is_empty()
                    && !m.aggregation.is_empty(),
                "invalid metric {}",
                m.id
            );
        }
        Ok(())
    }
}

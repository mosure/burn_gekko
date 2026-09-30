//! Paired readout differences within one checkpoint and one prediction population.
use crate::{
    benchmark::Row,
    statistics::{Interval, bootstrap_mean},
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadoutContrast {
    pub candidate: String,
    pub control: String,
}
#[derive(Debug, Serialize)]
pub struct PairedContrast {
    pub candidate: String,
    pub control: String,
    pub pairs: usize,
    pub aepe_gain: Interval,
    pub pck3_gain: Interval,
    pub interpretation: &'static str,
}

pub fn paired(rows: &[Row], contrast: &ReadoutContrast, eth3d: bool) -> Result<PairedContrast> {
    ensure!(
        contrast.candidate != contrast.control,
        "contrast requires distinct readouts"
    );
    let select = |method: &str| -> Result<BTreeMap<&str, &Row>> {
        let mut result = BTreeMap::new();
        for row in rows
            .iter()
            .filter(|r| r.method == method && (eth3d || r.group == "viewpoint"))
        {
            ensure!(
                result.insert(row.sample.as_str(), row).is_none(),
                "duplicate contrast pair"
            );
        }
        Ok(result)
    };
    let candidate = select(&contrast.candidate)?;
    let control = select(&contrast.control)?;
    ensure!(
        !candidate.is_empty() && candidate.keys().eq(control.keys()),
        "paired contrast populations differ"
    );
    let mut clusters = BTreeMap::<&str, BTreeMap<&str, Vec<[f64; 2]>>>::new();
    for (sample, c) in &candidate {
        let b = control[sample];
        ensure!(
            c.cluster == b.cluster && c.group == b.group && c.metrics.points == b.metrics.points,
            "paired label populations differ"
        );
        let gain = [
            b.metrics.aepe - c.metrics.aepe,
            c.metrics.pck3 - b.metrics.pck3,
        ];
        ensure!(
            gain.iter().all(|v| v.is_finite()),
            "nonfinite paired difference"
        );
        clusters
            .entry(&c.cluster)
            .or_default()
            .entry(&c.group)
            .or_default()
            .push(gain);
    }
    // Same group coverage in every cluster is required for equal scene/interval weighting.
    let expected = clusters
        .values()
        .next()
        .unwrap()
        .keys()
        .copied()
        .collect::<Vec<_>>();
    ensure!(
        clusters
            .values()
            .all(|v| v.keys().copied().collect::<Vec<_>>() == expected),
        "inconsistent cluster groups"
    );
    let values = |axis| {
        clusters
            .values()
            .map(|groups| {
                groups
                    .values()
                    .map(|pairs| pairs.iter().map(|v| v[axis]).sum::<f64>() / pairs.len() as f64)
                    .sum::<f64>()
                    / groups.len() as f64
            })
            .collect::<Vec<_>>()
    };
    Ok(PairedContrast {
        candidate: contrast.candidate.clone(),
        control: contrast.control.clone(),
        pairs: candidate.len(),
        aepe_gain: bootstrap_mean(&values(0), 719)?,
        pck3_gain: bootstrap_mean(&values(1), 719)?,
        interpretation: "Positive favors candidate: control minus candidate AEPE, candidate minus control PCK3. Pair means, then equal groups within each scene/sequence, then cluster bootstrap. One checkpoint; no training-seed uncertainty.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::correspondence;
    #[test]
    fn paired_weighting_does_not_let_large_scenes_dominate_and_rejects_missing_pairs() {
        let mut rows = Vec::new();
        for (scene, count, improvement) in [("small", 1, 1.), ("large", 9, 3.)] {
            for pair in 0..count {
                for (method, error) in [("candidate", 5.), ("control", 5. + improvement)] {
                    rows.push(Row {
                        sample: format!("{scene}-{pair}"),
                        cluster: scene.into(),
                        group: "3".into(),
                        method: method.into(),
                        metrics: correspondence(&[error, error]).unwrap(),
                    });
                }
            }
        }
        let c = ReadoutContrast {
            candidate: "candidate".into(),
            control: "control".into(),
        };
        let result = paired(&rows, &c, true).unwrap();
        assert_eq!(result.aepe_gain.mean, 2.);
        assert_eq!(result.pairs, 10);
        assert_eq!(result.aepe_gain.clusters, 2);
        rows.pop();
        assert!(paired(&rows, &c, true).is_err());
    }

    #[test]
    fn hpatches_contrast_excludes_illumination_and_preserves_gain_signs() {
        let mut rows = Vec::new();
        for (sequence, group, candidate, control) in [
            ("v_a", "viewpoint", 2., 6.),
            ("v_b", "viewpoint", 8., 5.),
            ("i_a", "illumination", 0., 1000.),
        ] {
            for (method, error) in [("candidate", candidate), ("control", control)] {
                rows.push(Row {
                    sample: format!("{sequence}-2"),
                    cluster: sequence.into(),
                    group: group.into(),
                    method: method.into(),
                    metrics: correspondence(&[error]).unwrap(),
                });
            }
        }
        let contrast = ReadoutContrast {
            candidate: "candidate".into(),
            control: "control".into(),
        };
        let result = paired(&rows, &contrast, false).unwrap();
        assert_eq!(result.pairs, 2);
        assert_eq!(result.aepe_gain.mean, 0.5);
        assert_eq!(result.pck3_gain.mean, 0.5);
        assert_eq!(result.aepe_gain.clusters, 2);
        rows[0].metrics.points += 1;
        assert!(paired(&rows, &contrast, false).is_err());
    }
}

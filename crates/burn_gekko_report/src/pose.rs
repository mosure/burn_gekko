//! Recheck camera-probe summaries and draw predicted correspondences without GT filtering.
use crate::{
    artifact::{Source, record},
    correspondence::{CorrespondenceFigure, line, point},
    figures::floats,
};
use anyhow::{Context, Result, ensure};
use burn_gekko_eval::pose::benchmark::{PoseReport, summarize};
use image::{Rgb, RgbImage};
use serde_json::{Value, json};
use std::path::Path;

fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x
            .as_f64()
            .zip(y.as_f64())
            .is_some_and(|(x, y)| (x - y).abs() <= 1e-10),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w)))
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same(x, y))
        }
        _ => a == b,
    }
}
pub fn verify(p: &PoseReport) -> Result<()> {
    burn_gekko_data::real_views::descriptor_grid(p.image_size)?;
    ensure!(
        p.inputs.len() == 4
            && [
                (&p.config.images, &p.config.images_sha256),
                (&p.config.labels, &p.config.labels_sha256),
                (&p.config.predictions, &p.config.predictions_sha256),
                (&p.config.provenance, &p.config.provenance_sha256),
            ]
            .into_iter()
            .all(|(path, sha)| p.inputs.get(path) == Some(sha)),
        "incomplete pose source closure"
    );
    ensure!(
        matches!(p.evaluation_use.as_str(), "held_out" | "development"),
        "missing pose evaluation role"
    );
    let expected = summarize(&p.rows, &p.config.methods)?;
    ensure!(
        same(&json!(expected), &json!(p.methods)),
        "pose summary differs from complete rows"
    );
    let candidate = expected
        .get("spatial_residual_conditional_local")
        .context("pose candidate")?;
    ensure!(
        p.contrasts.len() + 1 == expected.len(),
        "incomplete pose controls"
    );
    let mut controls = std::collections::BTreeSet::new();
    for c in &p.contrasts {
        let name = c["control"].as_str().context("pose control")?;
        ensure!(
            controls.insert(name)
                && name != "spatial_residual_conditional_local"
                && c["candidate"] == "spatial_residual_conditional_local",
            "invalid pose contrast"
        );
        let control = expected.get(name).context("unknown pose control")?;
        let gain = candidate.macro_pose_auc_10.context("candidate pose AUC")?
            - control.macro_pose_auc_10.context("control pose AUC")?;
        ensure!(
            (c["macro_auc10_gain"].as_f64().context("pose AUC gain")? - gain).abs() < 1e-10,
            "pose gain differs from rows"
        );
        let gains = candidate
            .sequences
            .iter()
            .map(|(seq, s)| {
                Ok((
                    seq.clone(),
                    s.pose_auc_10.context("sequence pose AUC")?
                        - control
                            .sequences
                            .get(seq)
                            .context("missing control sequence")?
                            .pose_auc_10
                            .context("sequence control AUC")?,
                ))
            })
            .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
        ensure!(
            same(&json!(gains), &c["sequence_auc10_gains"]),
            "pose sequence gain mismatch"
        );
        let passed = gains.values().all(|v| *v > 0.)
            && candidate.macro_success_fraction >= control.macro_success_fraction;
        ensure!(c["passed"] == passed, "pose gate verdict differs from rows");
    }
    Ok(())
}

/// Shared sequence rows for HTML/PDF. Keep solver success distinct from accuracy.
pub fn sequence_rows(p: &PoseReport) -> Vec<[String; 7]> {
    let mut rows = Vec::new();
    for (method, m) in &p.methods {
        let label = if method.starts_with("spatial_residual") {
            "Fusion"
        } else if method.starts_with("spatial_self") {
            "Same-image"
        } else {
            "Encoder"
        };
        for (name, s) in &m.sequences {
            let sequence = name
                .replace("long_office_household", "Office")
                .replace("structure_texture_far", "Structure far")
                .replace("structure_texture_near", "Structure near");
            rows.push([
                sequence,
                label.into(),
                format!("{}/{}", s.successes, s.pairs),
                format!("{:.2}", s.mean_rotation_degrees),
                s.mean_translation_degrees
                    .map(|v| format!("{v:.2}"))
                    .unwrap_or_else(|| "unavailable".into()),
                s.pose_auc_10
                    .map(|v| format!("{:.2}", 100. * v))
                    .unwrap_or_else(|| "unavailable".into()),
                s.excluded_low_baseline.to_string(),
            ]);
        }
    }
    rows
}

pub fn build(
    p: &PoseReport,
    out: &Path,
    sources: &mut Vec<Source>,
) -> Result<Vec<CorrespondenceFigure>> {
    let mut figures = Vec::new();
    let size = p.image_size;
    for (index, e) in p.examples.iter().enumerate() {
        let mut image = RgbImage::from_pixel(528, 256, Rgb([235, 240, 244]));
        for (role, offset) in [("target", 0), ("reference", 272)] {
            let info = &e["images"][role];
            let path = Path::new(info["file"].as_str().context("pose RGB")?);
            ensure!(
                record(path, sources)? == info["sha256"].as_str().context("pose RGB hash")?,
                "pose visual changed"
            );
            let values = floats(path, sources)?;
            ensure!(values.len() == size * size * 3, "pose RGB shape");
            let tile = RgbImage::from_raw(
                size as u32,
                size as u32,
                values
                    .iter()
                    .map(|v| (v.clamp(0., 1.) * 255.).round() as u8)
                    .collect(),
            )
            .unwrap();
            let tile = if size == 256 {
                tile
            } else {
                image::imageops::resize(&tile, 256, 256, image::imageops::FilterType::Triangle)
            };
            image::imageops::replace(&mut image, &tile, offset, 0);
        }
        let xy = |v: &Value, role: &str, offset: f64| -> Result<[f64; 2]> {
            let hw = &e["images"][role]["original_hw"];
            Ok([
                offset
                    + (v[0].as_f64().context("x")? + 0.5) * 256.
                        / hw[1].as_f64().context("width")?
                    - 0.5,
                (v[1].as_f64().context("y")? + 0.5) * 256. / hw[0].as_f64().context("height")?
                    - 0.5,
            ])
        };
        for v in e["matches"].as_array().context("pose visual matches")? {
            let a = xy(&v["target"], "target", 0.)?;
            let b = xy(&v["reference"], "reference", 272.)?;
            let color = if v["ransac_inlier"] == true {
                Rgb([24, 197, 158])
            } else {
                Rgb([244, 112, 66])
            };
            line(&mut image, a, b, color);
            point(&mut image, a[0], a[1], color);
            point(&mut image, b[0], b[1], color);
        }
        let file = format!("media/pose-{index:02}.png");
        image.save(out.join(&file))?;
        let angle = e["translation_degrees"]
            .as_f64()
            .map(|v| format!("{v:.2} degrees"))
            .unwrap_or_else(|| "excluded: baseline below 1 cm".into());
        let status = if e["success"] == true {
            "solver returned a pose".into()
        } else {
            format!("failed: {}", e["failure"].as_str().unwrap_or("unknown"))
        };
        figures.push(CorrespondenceFigure {title:format!("TUM camera motion / {}",e["pair"]["id"].as_str().context("pose pair id")?),file,
            caption:format!("Target left; reference right. Model RGB input {size}px; display thumbnails 256px. Predicted mutual correspondences: green marks RANSAC inliers, orange rejected matches. Inlier agreement is not ground-truth correctness. {status}; rotation error {:.2} degrees, signed translation-direction error {angle}. Known intrinsics enter the CPU solver only. {}.",e["rotation_degrees"].as_f64().context("rotation error")?,e["selection"].as_str().context("example selection")?)});
    }
    Ok(figures)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_gekko_eval::pose::{
        benchmark::{PoseRow, PoseScoreConfig},
        solver::{PoseFit, RelativePose, SolverConfig},
    };
    use std::collections::BTreeMap;
    #[test]
    fn camera_probe_rejects_summary_gate_and_source_tampering() {
        let names = vec![
            "spatial_residual_conditional_local".to_owned(),
            "spatial_self_conditional_local".to_owned(),
            "student_l06_centered_conditional_local".to_owned(),
        ];
        let rows = names
            .iter()
            .enumerate()
            .map(|(i, name)| PoseRow {
                pair: "pair".into(),
                sequence: "sequence".into(),
                interval: 15,
                method: name.clone(),
                mutual_matches: 16,
                fit: PoseFit {
                    pose: Some(RelativePose {
                        rotation: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                        translation: [1., 0., 0.],
                    }),
                    failure: None,
                    trials: 64,
                    inliers: vec![true; 16],
                    positive_depth_points: 16,
                },
                rotation_degrees: 1. + i as f64,
                translation_degrees: Some(1. + i as f64),
                pose_degrees: Some(1. + i as f64),
                baseline_meters: 0.1,
            })
            .collect::<Vec<_>>();
        let methods = summarize(&rows, &names).unwrap();
        let contrasts=names[1..].iter().map(|name|{
            let gain=methods[&names[0]].macro_pose_auc_10.unwrap()-methods[name].macro_pose_auc_10.unwrap();
            json!({"candidate":names[0],"control":name,"macro_auc10_gain":gain,"sequence_auc10_gains":{"sequence":gain},"passed":true})
        }).collect();
        let config = PoseScoreConfig {
            images: "images".into(),
            images_sha256: "a".into(),
            labels: "labels".into(),
            labels_sha256: "b".into(),
            predictions: "predictions".into(),
            predictions_sha256: "c".into(),
            provenance: "provenance".into(),
            provenance_sha256: "d".into(),
            checkpoint_sha256: "checkpoint".into(),
            methods: names,
            threshold_original_pixels: 3.,
            minimum_baseline_meters: 0.01,
            solver: SolverConfig {
                max_trials: 2048,
                min_trials: 64,
                confidence: 0.999,
                threshold: 0.005,
                min_inliers: 12,
                seed: 1,
            },
            output: "score".into(),
        };
        let inputs = BTreeMap::from([
            ("images".into(), "a".into()),
            ("labels".into(), "b".into()),
            ("predictions".into(), "c".into()),
            ("provenance".into(), "d".into()),
        ]);
        let mut report = PoseReport {
            schema: 1,
            image_size: 256,
            checkpoint_sha256: "checkpoint".into(),
            evaluation_use: "held_out".into(),
            protocol: "fixture".into(),
            config,
            inputs,
            rows,
            methods,
            contrasts,
            examples: vec![],
        };
        verify(&report).unwrap();
        report.contrasts[0]["passed"] = json!(false);
        assert!(verify(&report).unwrap_err().to_string().contains("verdict"));
        report.contrasts[0]["passed"] = json!(true);
        report
            .methods
            .get_mut("spatial_residual_conditional_local")
            .unwrap()
            .macro_pose_auc_10 = Some(1.);
        assert!(verify(&report).unwrap_err().to_string().contains("summary"));
        report.inputs.clear();
        assert!(
            verify(&report)
                .unwrap_err()
                .to_string()
                .contains("source closure")
        );
    }
}

//! Matched renderer-supervised screening; independent of external benchmark outcomes.
use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewArm {
    pub summary: Input,
    pub geometry: Input,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionConfig {
    pub control: ViewArm,
    pub candidate: ViewArm,
    pub updates: u64,
    pub expected_weight: f64,
    pub output: PathBuf,
}
pub fn gates(control: &Measurements, candidate: &Measurements) -> Value {
    let mse = candidate.cross_mse <= control.cross_mse * 1.01;
    let references = candidate.cross_mse < candidate.monocular_mse;
    let error = candidate.warp_mean_pixel_error <= control.warp_mean_pixel_error * 0.95;
    let precision = candidate.warp_pck8 >= control.warp_pck8;
    json!({"latent_mse_within_one_percent":mse,"references_help":references,"at_least_five_percent_lower_view_error":error,"nondecreasing_view_pck8":precision,"passed":mse && references && error && precision})
}
pub fn select(c: &SelectionConfig) -> Result<Value> {
    ensure!(
        !c.output.exists() && c.updates == 384 && c.expected_weight == 0.1,
        "unregistered geometry screen or existing output"
    );
    let mut sources = BTreeMap::new();
    let mut load = |a: &ViewArm| -> Result<(Evidence, Value)> {
        let summary = pinned(&a.summary, &mut sources)?;
        let geometry = pinned(&a.geometry, &mut sources)?;
        ensure!(
            geometry["task"] == "renderer_viewpoint_correspondence",
            "not actual viewpoint scoring"
        );
        let evidence = arm(
            &Arm {
                summary: Input {
                    path: a.summary.path.clone(),
                    sha256: a.summary.sha256.clone(),
                },
                warp: Input {
                    path: a.geometry.path.clone(),
                    sha256: a.geometry.sha256.clone(),
                },
            },
            2,
            c.updates,
            &mut sources,
        )?;
        Ok((evidence, summary))
    };
    let (control, control_summary) = load(&c.control)?;
    let (mut candidate, candidate_summary) = load(&c.candidate)?;
    ensure!(
        control_summary["source_sha256"].is_string()
            && control_summary["source_sha256"] == candidate_summary["source_sha256"]
            && control_summary["teacher_id"] == candidate_summary["teacher_id"],
        "source/teacher mismatch"
    );
    ensure!(
        control.recipe.get("view_geometry").is_none(),
        "control uses geometry objective"
    );
    let objective = candidate
        .recipe
        .as_object_mut()
        .unwrap()
        .remove("view_geometry")
        .context("candidate lacks geometry objective")?;
    ensure!(
        number(&objective["weight"])? == c.expected_weight
            && number(&objective["temperature"])? == 0.07,
        "objective differs from registration"
    );
    ensure!(
        candidate_summary["view_geometry"]["config"] == objective,
        "geometry provenance differs"
    );
    for s in [&control_summary, &candidate_summary] {
        ensure!(
            s["parameter_probes"]["preservation_anchor_qkv_max_abs_delta"] == json!([0., 0.]),
            "fixed anchor changed"
        );
    }
    ensure!(
        number(&candidate_summary["scalar_windows"]["view_geometry_nll"]["last_mean"])? > 0.
            && number(
                &candidate_summary["scalar_windows"]["view_geometry_valid_fraction"]["last_mean"]
            )? > 0.,
        "inactive geometry objective"
    );
    let geometry = pinned(&c.candidate.geometry, &mut sources)?;
    let baseline = pinned(&c.control.geometry, &mut sources)?;
    ensure!(
        geometry["cache_manifest_sha256"] == baseline["cache_manifest_sha256"]
            && geometry["cache_manifest_sha256"]
                == candidate_summary["view_geometry"]["cache_manifest_sha256"],
        "geometry target provenance differs"
    );
    matched(&control, &candidate)?;
    let decision = gates(&control.metrics, &candidate.metrics);
    let passed = decision["passed"] == true;
    let measured = |m: &Measurements| json!({"latent_mse":m.cross_mse,"monocular_mse":m.monocular_mse,"view_aepe_pixels":m.warp_mean_pixel_error,"view_pck8":m.warp_pck8});
    let result = json!({"schema":1,"updates_per_arm":c.updates,"selected_arm":if passed {"geometry"} else {"control"},"selected_checkpoint_sha256":if passed {&candidate.checkpoint} else {&control.checkpoint},"control":measured(&control.metrics),"candidate":measured(&candidate.metrics),"gates":decision,"sources":sources,"scope":"Fixed short synthetic-development screen. Matched updates, recipes, sample order, masks and visible validation queries; extra geometry decoder work is measured, not called equal compute. External outcomes cannot select the arm. No long-run or SOTA qualification."});
    write_json(&c.output, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometric_gain_cannot_hide_completion_or_precision_regression() {
        let m = |mse, error, pck| Measurements {
            cross_mse: mse,
            monocular_mse: 0.25,
            warp_mean_pixel_error: error,
            warp_pck8: pck,
        };
        let control = m(0.2, 20., 0.3);
        assert_eq!(gates(&control, &m(0.201, 18., 0.31))["passed"], true);
        for candidate in [
            m(0.203, 18., 0.31),
            m(0.201, 19.5, 0.31),
            m(0.201, 18., 0.29),
        ] {
            assert_eq!(gates(&control, &candidate)["passed"], false);
        }
    }
}

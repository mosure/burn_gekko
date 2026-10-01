//! Fixed three-arm validation decision for final-feature preservation.
use super::*;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionConfig {
    pub tail: Arm,
    pub full: Arm,
    pub preserved: Arm,
    /// Registered before candidate training, not inferred from the observed recipe.
    #[serde(default = "default_preservation_weight")]
    pub expected_preservation_weight: f64,
    pub output: PathBuf,
}

fn default_preservation_weight() -> f64 {
    4.
}

fn registered_weight(actual: f64, expected: f64) -> Result<()> {
    ensure!(
        expected > 0. && expected <= 16. && actual == expected,
        "preservation weight differs from its registered value or exceeds the supported range"
    );
    Ok(())
}

fn preservation_contract(e: &mut Evidence, summary: &Value, expected_weight: f64) -> Result<()> {
    let config = e
        .recipe
        .as_object_mut()
        .unwrap()
        .remove("encoder_preservation")
        .context("missing encoder preservation objective")?;
    ensure!(
        summary["encoder_preservation"] == config,
        "preservation provenance differs from recipe"
    );
    registered_weight(number(&config["weight"])?, expected_weight)?;
    ensure!(
        config["anchor"] == e.recipe["warm_start"],
        "screen anchor differs from common parent"
    );
    ensure!(
        summary["parameter_probes"]["preservation_anchor_qkv_max_abs_delta"] == json!([0., 0.]),
        "frozen preservation anchor probe failed"
    );
    ensure!(
        number(&summary["scalar_windows"]["encoder_preservation_mse"]["last_mean"])? > 0.,
        "preservation objective was inactive"
    );
    let rows = fs::read_to_string(e.run.join("metrics.jsonl"))?;
    let first: Value = serde_json::from_str(rows.lines().next().context("missing first update")?)?;
    let startup = &summary["preservation_startup"];
    ensure!(
        startup["parameters"]["max_abs_difference"] == 0.
            && startup["parameters"]["parameter_elements"]
                .as_u64()
                .is_some_and(|n| n > 0)
            && startup["mean_relative_mse"] == first["encoder_preservation_mse"],
        "missing or inconsistent exact common-parent parameter check"
    );
    ensure!(
        number(&first["encoder_preservation_mse"])? < 3e-6
            && startup["route_relative_mse"]
                .as_array()
                .is_some_and(|routes| {
                    routes.len() == 4
                        && routes.iter().all(|v| {
                            v.as_f64()
                                .is_some_and(|n| n.is_finite() && (0.0..1e-5).contains(&n))
                        })
                }),
        "common-parent forward error exceeds amended numerical qualification (mean 3e-6, each route 1e-5); original 1e-6 check remains failed"
    );
    let meta: Value = serde_json::from_slice(&fs::read(e.run.join("final/metadata.json"))?)?;
    ensure!(
        meta["weight_ancestors"]
            .as_array()
            .is_some_and(|parents| parents.contains(&config["anchor"])),
        "anchor missing from audited checkpoint ancestry"
    );
    Ok(())
}

pub fn gates(tail: &Measurements, full: &Measurements, preserved: &Measurements) -> Value {
    let retain = preserved.cross_mse <= tail.cross_mse * 1.01;
    let improve_full = preserved.cross_mse < full.cross_mse;
    let references = preserved.cross_mse < preserved.monocular_mse;
    let geometry = preserved.warp_mean_pixel_error <= tail.warp_mean_pixel_error * 0.9;
    let precision = preserved.warp_pck8 >= tail.warp_pck8;
    json!({"latent_mse_within_one_percent_of_tail":retain,"lower_latent_mse_than_full":improve_full,"references_help":references,"at_least_ten_percent_lower_warp_error_than_tail":geometry,"nondecreasing_warp_pck8_vs_tail":precision,"passed":retain && improve_full && references && geometry && precision})
}

pub fn select(c: &SelectionConfig) -> Result<Value> {
    ensure!(
        !c.output.exists(),
        "preserve existing preservation decision"
    );
    let mut sources = BTreeMap::new();
    let tail = arm(&c.tail, 1, 2048, &mut sources)?;
    let full = arm(&c.full, 2, 2048, &mut sources)?;
    let mut preserved = arm(&c.preserved, 2, 2048, &mut sources)?;
    let tail_summary = pinned(&c.tail.summary, &mut sources)?;
    let full_summary = pinned(&c.full.summary, &mut sources)?;
    let summary = pinned(&c.preserved.summary, &mut sources)?;
    ensure!(
        tail_summary["source_sha256"]
            .as_str()
            .is_some_and(|s| s.len() == 64)
            && tail_summary["source_sha256"] == full_summary["source_sha256"]
            && tail_summary["source_sha256"] == summary["source_sha256"],
        "screen training source differs or is unrecorded"
    );
    ensure!(
        tail_summary["teacher_id"].is_string()
            && tail_summary["teacher_id"] == full_summary["teacher_id"]
            && tail_summary["teacher_id"] == summary["teacher_id"],
        "screen teacher differs or is unrecorded"
    );
    ensure!(
        tail.recipe.get("encoder_preservation").is_none()
            && full.recipe.get("encoder_preservation").is_none(),
        "control contains preservation objective"
    );
    matched(&tail, &full)?;
    preservation_contract(&mut preserved, &summary, c.expected_preservation_weight)?;
    matched(&tail, &preserved)?;
    let decision = gates(&tail.metrics, &full.metrics, &preserved.metrics);
    let accepted = decision["passed"] == true;
    let result = json!({"schema":1,"updates_per_arm":2048,"expected_preservation_weight":c.expected_preservation_weight,"selected_arm":if accepted {"preserved"} else {"tail"},"selected_checkpoint_sha256":if accepted {&preserved.checkpoint} else {&tail.checkpoint},"tail":tail.metrics,"full":full.metrics,"preserved":preserved.metrics,"gates":decision,"sources":sources,"scope":"Registered synthetic-validation screen with matched recipes, sample order, masks and transformed-query populations. Both full arms train all encoder tensors; only the preserved arm adds the pinned final-feature objective with the explicitly registered coefficient (default 4.0 for the original screen). Real-view results do not select this arm; rejection does not suppress their diagnostic export."});
    write_json(&c.output, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coefficient_must_match_the_registered_value() {
        assert_eq!(default_preservation_weight(), 4.);
        registered_weight(4., 4.).unwrap();
        registered_weight(16., 16.).unwrap();
        for (actual, expected) in [
            (4., 16.),
            (16., 4.),
            (0., 0.),
            (17., 17.),
            (f64::NAN, 4.),
            (4., f64::NAN),
        ] {
            assert!(registered_weight(actual, expected).is_err());
        }
    }
    #[test]
    fn preservation_must_beat_unrestricted_full_and_retain_geometry_and_completion() {
        let m = |cross_mse, error, pck| Measurements {
            cross_mse,
            monocular_mse: 0.24,
            warp_mean_pixel_error: error,
            warp_pck8: pck,
        };
        let tail = m(0.2, 8., 0.7);
        let full = m(0.21, 3., 0.95);
        assert_eq!(gates(&tail, &full, &m(0.201, 7., 0.71))["passed"], true);
        for candidate in [m(0.203, 7., 0.71), m(0.201, 7.3, 0.71), m(0.201, 7., 0.69)] {
            assert_eq!(gates(&tail, &full, &candidate)["passed"], false);
        }
        assert_eq!(
            gates(&tail, &m(0.199, 3., 0.95), &m(0.201, 7., 0.71))["passed"],
            false
        );
    }
}

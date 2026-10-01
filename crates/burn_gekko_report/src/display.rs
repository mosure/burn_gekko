//! Human-readable labels; metric identities and stored numeric units remain unchanged.
use burn_gekko_eval::schema::{Capability, Metric};

/// Keep stored method IDs intact while making solver panels readable in print.
pub fn head_labels(capability: &mut Capability) {
    if !capability.id.starts_with("pose_solver_stability_") {
        return;
    }
    for (method, label) in [
        ("spatial_residual_conditional_local", "Pair-conditioned"),
        ("spatial_self_conditional_local", "Same-image control"),
        (
            "student_l06_centered_conditional_local",
            "Encoder block 6 control",
        ),
    ] {
        for metric in &mut capability.metrics {
            metric.label = metric.label.replace(method, label);
        }
        for limitation in &mut capability.limitations {
            *limitation = limitation.replace(method, label);
        }
    }
}

pub fn view_geometry(training: &serde_json::Value) -> Option<String> {
    let config = &training["config"]["view_geometry"];
    if !config.is_object() {
        return None;
    }
    let loss = &training["scalar_windows"]["view_geometry_nll"];
    let valid = &training["scalar_windows"]["view_geometry_valid_fraction"];
    Some(format!(
        "Renderer-supervised correspondence is a training-only auxiliary (weight {}, temperature {}). First versus last {}-update mean NLL: {:.3} to {:.3} nats; usable query fraction: {:.1}% to {:.1}%. These are changing-minibatch diagnostics. Geometry and camera labels are absent from RGB inference.",
        config["weight"],
        config["temperature"],
        loss["window_updates"],
        loss["first_mean"].as_f64()?,
        loss["last_mean"].as_f64()?,
        100. * valid["first_mean"].as_f64()?,
        100. * valid["last_mean"].as_f64()?
    ))
}

pub fn encoder_preservation(training: &serde_json::Value) -> Option<String> {
    let weight = training["config"]["encoder_preservation"]["weight"].as_f64()?;
    let windows = &training["scalar_windows"]["encoder_preservation_mse"];
    let first = windows["first_mean"].as_f64()?.sqrt() * 100.;
    let last = windows["last_mean"].as_f64()?.sqrt() * 100.;
    let startup = training["preservation_startup"]["mean_relative_mse"].as_f64().map(|mse|format!(" Before any update, identical encoder parameters produce {:.3}% RMS forward discrepancy; this numerical floor is included in the drift values, not subtracted.",mse.sqrt()*100.)).unwrap_or_default();
    Some(format!(
        "Final encoder features are anchored to a pinned frozen ancestor with loss weight {weight:.1}. Training feature RMS drift is {first:.2}% in the first and {last:.2}% in the last {}-update window (100 times the square root of mean per-image, energy-normalized feature MSE, equally weighted over sparse/full target and reference routes). Intermediate geometric features have no direct preservation target. This is a training regularization diagnostic, not RGB error or independent accuracy. The frozen anchor is absent from inference.{startup}",
        windows["window_updates"]
    ))
}

pub fn encoder_stages(training: &serde_json::Value) -> Option<String> {
    let stages = training["encoder_stages"].as_object()?;
    if stages.is_empty() {
        return None;
    }
    let parts = stages
        .iter()
        .map(|(stage, value)| {
            let label = match stage.as_str() {
                "0" => "encoder frozen",
                "1" => "last two encoder blocks and output norms trainable",
                _ => "full encoder trainable",
            };
            format!(
                "{} updates with {label} (steps {}-{})",
                value["updates"], value["first_step"], value["last_step"]
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    Some(format!(
        "Verified training stages: {parts}. Logged encoder gradient counts are checked against each stage. These counts describe optimization, not evidence of improved transfer."
    ))
}

pub fn readout(method: &str) -> String {
    if let Some(base) = method.strip_suffix("_local") {
        return format!("{} + local refinement", readout(base));
    }
    match method {
        "fused_decoder" => "Fusion decoder features".into(),
        "conditional_decoder" => "Fusion decoder, conditional matching".into(),
        "reciprocal_log_probability" => "Reciprocal fusion attention".into(),
        "spatial_residual" => "Spatial residual descriptors".into(),
        "spatial_residual_conditional" => "Pair-conditioned descriptor".into(),
        "spatial_pair" => "Pair-conditioned descriptor".into(),
        "spatial_self" | "spatial_self_conditional" => "Same-image conditioned control".into(),
        "spatial_encoder" => "Centered encoder control".into(),
        _ => {
            if let Some(rest) = method.strip_prefix("student_l")
                && let Some((block, suffix)) = rest.split_once('_')
                && let Ok(block) = block.parse::<usize>()
            {
                return format!(
                    "Encoder block {block}{}",
                    if suffix.ends_with("conditional") {
                        " control"
                    } else {
                        ", centered features"
                    }
                );
            }
            method.replace('_', " ")
        }
    }
}

pub fn value(metric: &Metric) -> (String, &str) {
    if metric.unit == "fraction" {
        (format!("{:.2}", 100. * metric.value), "%")
    } else if matches!(metric.unit.as_str(), "dB" | "pixels" | "degrees") {
        (format!("{:.2}", metric.value), &metric.unit)
    } else {
        (format!("{:.4}", metric.value), &metric.unit)
    }
}

/// Shared explanations for the HTML page and PDF; actual formulas stay in Rust metrics.
pub const METRIC_GUIDE: &[(&str, &str)] = &[
    (
        "Calibrated camera-motion probe",
        "A CPU geometric solver estimates motion from the model's matches and known intrinsics. Rotation and signed translation-direction errors use degrees; lower is better. Pose error is the larger angle. Pose AUC summarizes recall up to its angular threshold, displayed as a percentage; higher is better. Failed fits count as 180 degrees. Solver success means an estimate was returned, not that it was correct. This is separate from learned camera or focal-length prediction.",
    ),
    (
        "Feature signal / error (dB)",
        "10 log10(mean squared teacher feature / prediction MSE), measured on hidden patches and averaged over target views. Higher is better; +3.01 dB means half the error at the same signal power. It is not RGB PSNR and has no universal image-quality threshold.",
    ),
    (
        "RGB PSNR (dB)",
        "10 log10(RGB data range squared / RGB MSE). Reported only for an evaluated RGB decoder with a declared range, color space and mask. Colored feature plots cannot be scored as reconstructed images.",
    ),
    (
        "Feature MSE and cosine",
        "MSE is mean squared feature error: zero is exact. Cosine measures feature-direction agreement: one is perfect alignment. A high cosine can coexist with lost spatial detail, so also inspect the variation ratio and shared-color feature maps.",
    ),
    (
        "Reference benefit (%)",
        "100 times (1 - cross-view MSE / references-disabled MSE), using population means from the same checkpoint and targets. Positive means the reference images reduce error. Negative means they hurt.",
    ),
    (
        "Spatial variation retained (%)",
        "Predicted spatial feature variance divided by teacher variance. 100% matches the teacher's amount of variation; lower values indicate smoothing. More than 100% is not automatically better, and matched variance does not prove correct structure.",
    ),
    (
        "Mean match error (pixels)",
        "AEPE: average distance between a predicted correspondence and its true position. Lower is better. HPatches uses the declared 240 by 240 scoring frame; ETH3D uses original image pixels. Neither is measured in the displayed thumbnail's pixels.",
    ),
    (
        "Within 3 pixels (%)",
        "PCK3: percentage of labeled matches whose error is at most three pixels in the declared scoring frame. Higher is better; 100% is perfect. A gain of 1 percentage point means one additional match per hundred after the stated macro weighting.",
    ),
    (
        "Co-visibility ranking",
        "AUROC compares visible and non-visible patches: 0.5 is chance and 1 is perfect. Average precision (AP) also depends on how many patches are visible, so read it alongside the positive-label fraction. These are ranking measures, not calibrated probabilities.",
    ),
    (
        "Known-transform loss (nats)",
        "Negative log likelihood of the soft correspondence targets, using natural logarithms. Lower is better. Its scale depends on the grid, temperature and label distribution; compare it only within that fixed objective.",
    ),
    (
        "Camera and future heads",
        "Rotation and translation-direction errors are angles in degrees (lower is better). Focal-length relative error can be shown as a percentage. Pose AUC reports accuracy over an angular threshold range. No camera, depth or RGB capability is claimed until that head is trained and evaluated.",
    ),
];

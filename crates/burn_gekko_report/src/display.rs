//! Human-readable labels; metric identities and stored numeric units remain unchanged.
use burn_gekko_eval::schema::Metric;

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
    match method {
        "fused_decoder" => "Fusion decoder features".into(),
        "conditional_decoder" => "Fusion decoder, conditional matching".into(),
        "reciprocal_log_probability" => "Reciprocal fusion attention".into(),
        "spatial_residual" => "Spatial residual descriptors".into(),
        "spatial_residual_conditional" => "Spatial residual, conditional matching".into(),
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
                        ", conditional matching"
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
    } else {
        (format!("{:.4}", metric.value), &metric.unit)
    }
}

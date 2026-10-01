use crate::{artifact::Report, experiment::Experiment, figures::Sample};
use anyhow::Result;
use burn_gekko_eval::schema::CapabilityStatus;
use std::{fs, path::Path};
pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
pub fn write(
    e: &Experiment,
    r: &Report,
    samples: &[Sample],
    matches: &[crate::correspondence::CorrespondenceFigure],
    out: &Path,
    pdf: bool,
) -> Result<()> {
    let mut capabilities = String::new();
    for c in &r.capabilities {
        let status = match c.status {
            CapabilityStatus::Evaluated => "Evaluated",
            CapabilityStatus::NotEvaluated => "Not evaluated",
            CapabilityStatus::NotTrained => "Not trained",
            CapabilityStatus::NotImplemented => "Not implemented",
        };
        capabilities.push_str(&format!("<article class='capability'><div class='cap-title'><h3>{}</h3><span class='status {}'>{status}</span></div><p>{}</p>",escape(&c.label),if c.status==CapabilityStatus::Evaluated{"measured"}else{"pending"},escape(&c.protocol)));
        if !c.metrics.is_empty() {
            capabilities.push_str("<div class='table-wrap'><table><thead><tr><th>Metric</th><th>Value</th><th>Unit / population</th></tr></thead><tbody>");
            for m in &c.metrics {
                let (value, unit) = crate::display::value(m);
                capabilities.push_str(&format!("<tr><td>{}</td><td class='numeric'>{}</td><td>{} · n = {}<small>{}</small></td></tr>",escape(&m.label),value,escape(unit),m.samples,escape(&m.aggregation)));
            }
            capabilities.push_str("</tbody></table></div>");
        }
        for limitation in &c.limitations {
            capabilities.push_str(&format!("<p class='note'>{}</p>", escape(limitation)));
        }
        capabilities.push_str("</article>");
    }
    if let Some(pose) = &r.calibrated_pose {
        capabilities.push_str("<article class='capability'><h3>Camera motion by sequence</h3><p>Solver success counts estimates returned, including inaccurate ones. Rotation and translation-direction errors include failed fits at 180°. AUC@10 is shown as a percentage. Low-baseline exclusions affect translation and pose only.</p><div class='table-wrap'><table><thead><tr><th>Sequence</th><th>Readout</th><th>Solved / all</th><th>Rotation °</th><th>Translation °</th><th>AUC@10 %</th><th>Excluded</th></tr></thead><tbody>");
        for row in crate::pose::sequence_rows(pose) {
            capabilities.push_str("<tr>");
            for cell in row {
                capabilities.push_str(&format!("<td>{}</td>", escape(&cell)));
            }
            capabilities.push_str("</tr>");
        }
        capabilities.push_str("</tbody></table></div><p class='note'>TUM RGB-D data: <a href='https://cvg.cit.tum.de/data/datasets/rgbd-dataset'>Sturm et al.; CC BY 4.0</a>. These three sequences share one acquisition environment. Prior exposure in the pretrained encoder's data is not established.</p></article>");
    }
    let first = &samples[0];
    let sample_snr = first
        .feature_snr_db
        .map(|x| format!("Feature signal/error {x:.2} dB · "))
        .unwrap_or_default();
    let mut gallery = String::new();
    for (i, (title, file)) in first.panels.iter().enumerate() {
        gallery.push_str(&format!("<figure><img id='panel-{i}' src='{}' alt='{}' width='256' height='256'><figcaption>{}</figcaption></figure>",escape(file),escape(title),escape(title)));
    }
    let options = samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            format!(
                "<option value='{i}'>Room {} · target view {}</option>",
                s.room_seed, s.target_view
            )
        })
        .collect::<String>();
    let limits = e
        .limitations
        .iter()
        .map(|s| format!("<li>{}</li>", escape(s)))
        .collect::<String>();
    let metrics = &r.latent;
    let mse = metrics["mean_cross_mse"].as_f64().unwrap();
    let reference_benefit = 100. * (1. - mse / metrics["mean_monocular_mse"].as_f64().unwrap());
    let rgb_psnr = r
        .capabilities
        .iter()
        .find(|c| c.id == "rgb_completion")
        .and_then(|c| c.metrics.iter().find(|m| m.id == "psnr"));
    let (primary_value, primary_label, primary_note) = if let Some(psnr) = rgb_psnr {
        (
            format!("{:.2} dB", psnr.value),
            "Hidden-pixel RGB PSNR ↑",
            "Attached head; separate development cohort",
        )
    } else if let Some(snr) = metrics["mean_feature_snr_db"].as_f64() {
        (
            format!("{snr:.2} dB"),
            "Feature signal / error ↑",
            "Hidden patches; not RGB PSNR",
        )
    } else {
        (
            format!("{mse:.4}"),
            "Hidden-token MSE ↓",
            "Fixed V-JEPA latent space",
        )
    };
    let guide = crate::display::METRIC_GUIDE
        .iter()
        .map(|(name, detail)| format!("<dt>{}</dt><dd>{}</dd>", escape(name), escape(detail)))
        .collect::<String>();
    let auc = metrics["learned_ri_covisibility"]["auroc"]
        .as_f64()
        .unwrap();
    let masked = 100. * r.training["config"]["mask_ratio"].as_f64().unwrap_or(0.9);
    let data = serde_json::to_string(samples)?
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026");
    let mut correspondence = String::new();
    let mut output_heads = String::new();
    if !matches.is_empty() {
        correspondence.push_str("<section><p class='eyebrow'>GEOMETRIC CORRESPONDENCE</p><h2>Where do the features match?</h2><p>Each caption identifies its protocol and colors. Matching panels show ground-truth locations after inference; camera-motion panels show predicted matches and solver inliers. An inlier is not necessarily a correct match. All examples use this checkpoint.</p><div class='match-grid'>");
        for m in matches.iter().filter(|m| !m.file.contains("output-head-")) {
            correspondence.push_str(&format!("<figure><h3>{}</h3><img src='{}' alt='Annotated target and reference matches for {}'><figcaption>{}</figcaption></figure>",escape(&m.title),escape(&m.file),escape(&m.title),escape(&m.caption)));
        }
        correspondence.push_str("</div></section>");
    }
    if e.output_heads.is_some() {
        output_heads.push_str("<section id='output-heads'><p class='eyebrow'>LEARNED OUTPUT HEADS</p><h2>RGB pixels and camera calibration.</h2><p>One attached head-training run on the frozen foundation. RGB completion uses sparse target features; camera prediction has its own dense RGB-pair route. Teacher pixels and camera labels are evaluation targets. Captions show hidden-pixel PSNR and angular calibration errors.</p><div class='match-grid'>");
        for m in matches.iter().filter(|m| m.file.contains("output-head-")) {
            output_heads.push_str(&format!(
                "<figure><h3>{}</h3><img src='{}' alt='{}'><figcaption>{}</figcaption></figure>",
                escape(&m.title),
                escape(&m.file),
                escape(&m.title),
                escape(&m.caption)
            ));
        }
        output_heads.push_str("</div></section>");
    }
    let efficiency = if let Some(v) = &r.efficiency {
        let batch = r.training["config"]["batch_size"].as_f64().unwrap_or(1.);
        let board = format!(
            "Mean observed board power {:.1} W; median device activity {}; peak training-process VRAM {}. Activity includes shared desktop work and does not measure SM occupancy.",
            v.observed_mean_board_power_w,
            v.median_device_gpu_percent
                .map(|x| format!("{x:.0}%"))
                .unwrap_or_else(|| "unavailable".into()),
            v.peak_process_vram_mib
                .map(|x| format!("{:.2} GiB", x / 1024.))
                .unwrap_or_else(|| "unavailable".into())
        );
        let stages = v
            .median_update_seconds_by_stage
            .iter()
            .map(|(stage, median)| {
                format!(
                    "<tr><td>{}</td><td>{:.3} s</td><td>{:.3} s</td><td>{:.1}</td></tr>",
                    escape(stage),
                    median,
                    v.p95_update_seconds_by_stage[stage],
                    batch / median
                )
            })
            .collect::<String>();
        format!(
            "<h3>Training efficiency</h3><p>{:.1} command minutes · {:.2} Wh observed board energy · {:.2} J per logged target · {:.1}% telemetry coverage.</p><p class='note'>{}</p><p>{board}</p><div class='table-wrap'><table><thead><tr><th>Encoder stage</th><th>Median update</th><th>95th percentile</th><th>Targets / second</th></tr></thead><tbody>{stages}</tbody></table></div><p class='note'>Stage 0 freezes the encoder; stage 1 adapts its final blocks; stage 2 adapts all blocks. Throughput is per-update training throughput, excluding validation/checkpoint overhead.</p>",
            v.command_seconds / 60.,
            v.observed_board_energy_wh,
            v.observed_board_joules_per_target,
            100. * v.telemetry_coverage,
            escape(&v.scope)
        )
    } else {
        "<p class='note'>No command-bound power telemetry supplied for this report.</p>".into()
    };
    let camera_note = if r
        .capabilities
        .iter()
        .any(|c| c.id == "camera" && c.status == CapabilityStatus::Evaluated)
    {
        "Camera output measurements appear in the capability table. Image-grid RoPE itself is not a calibrated 3D camera encoding."
    } else {
        "No evaluated camera head in this checkpoint. Image-grid RoPE is not a calibrated 3D camera encoding."
    };
    let html = format!(
        r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>{title}</title><meta name="description" content="One auditable Burn multi-view training experiment, with visual evaluations and explicit capability boundaries."><link rel="stylesheet" href="style.css"></head>
<body><header><a class="brand" href="#">burn_gekko<span> / research</span></a><nav aria-label="Sections"><a href="#explore">Explore</a><a href="#evidence">Metrics</a><a href="#method">Model</a><a href="#paper">Paper</a></nav><span class="draft">Research draft</span></header>
<main><section class="hero"><p class="eyebrow">MULTI-VIEW LEARNING · BUILT IN BURN</p><h1>Learn what<br>the other views reveal.</h1><p class="lead">{description}</p><p class="byline">{author} · {id} · one experiment, pinned training lineage</p><div class="actions"><a class="button" href="paper.pdf">Read the paper ↗</a><a class="button secondary" href="#explore">Explore predictions ↓</a></div><p class="note">Research prototype. State-of-the-art performance and sharp RGB reconstruction are not established.</p></section>
<section class="cards" aria-label="Measured capabilities"><article><strong>{primary_value}</strong><span>{primary_label}</span><small>{primary_note}</small></article><article><strong>{reference_benefit:.2}%</strong><span>Latent error reduction from references ↑</span><small>Same checkpoint, same target mask</small></article><article><strong>{auc:.3}</strong><span>Co-visibility AUROC ↑</span><small>Separate full-target RI branch</small></article><article><strong>{masked:.0}%</strong><span>Target patches hidden</span><small>References encoded independently</small></article></section>
<section id="explore"><p class="eyebrow">01 / CROSS-VIEW COMPLETION</p><h2>Same scene.<br>Different information.</h2><p class="section-intro">The model receives the sparse target and reference images. The full target and teacher features below are evaluation targets. Latent colors share a teacher-fitted projection and color scale; they do not show reconstructed RGB.</p><div class="controls"><label for="sample">Evaluation sample</label><select id="sample">{options}</select><span id="sample-metric">{sample_snr}MSE {sample_mse:.4} · cosine {sample_cos:.3}</span></div><div class="sample-grid">{gallery}</div><div class="legend"><span><i class="teal"></i> Visible in a reference</span><span><i class="coral"></i> Not visible</span><span><i class="gray"></i> Observed token / unknown label</span><span class="heat-legend">Error: 0 <i></i> 1+</span></div><p class="note">Deterministic evenly spaced selection from the exported cohort, including its endpoints. No selection by quality. Error, RI and visibility fraction use fixed 0–1 scales. Reference benefit is (monocular MSE − cross-view MSE) / monocular MSE, displayed from −1 (red: worse) through 0 (white) to +1 (teal: better). Values outside the display range saturate. RI uses the full target and is a regression score, not a calibrated visibility probability. The complete population, selection and projection are downloadable.</p><details><summary>How to read this result</summary><p>Compare the teacher and predicted latent patterns using their common colors. Lost variation indicates oversmoothing in representation space. Compare the sparse input with both references to judge how much shared structure is available. The displayed co-visibility labels score these evaluation images after prediction. Any use of training-room geometry is declared separately in the experiment architecture; geometry is never a model input.</p></details></section>
<section id="evidence"><p class="eyebrow">02 / MEASURED CAPABILITIES</p><h2>Every head has an evidence status.</h2><p>All measurements below bind to checkpoint <code>{short_hash}</code>. Blank capabilities remain explicit until their heads are trained and evaluated. Benchmark readouts use their declared coordinate systems; numbers from different protocols are not interchangeable.</p><details open class="metric-guide"><summary>How to read the numbers</summary><dl>{guide}</dl></details>{capabilities}<details><summary>Input isolation and within-model controls</summary><pre>{audit}</pre><p>Reference-count or monocular controls measure information use within this model. This page does not compare private model versions.</p></details></section>
<section id="method"><p class="eyebrow">03 / MODEL & TRAINING</p><h2>Independent views.<br>A shared fusion model.</h2><p>{architecture}</p><div class="pipeline" aria-label="Model pipeline"><div>Sparse target<br><small>original patch positions</small></div><b>→</b><div>Shared V-JEPA encoder<br><small>MIT weights + adaptation</small></div><b>→</b><div>Multi-view fusion<br><small>2D rotary attention</small></div><b>→</b><div>Task heads<br><small>{task_heads}</small></div></div><p>Reference views enter the shared encoder independently and join the fusion stage as a set. Dense teacher features enter the completion loss only. The RI branch sees the full target under its separate contract.</p><figure class="training"><img src="media/training.svg" alt="Selected-phase training MSE in 50-update means and scheduled validation probes"><figcaption>This phase through the selected checkpoint: training means over at most 50 updates, with scheduled validation when available. Validation is development data; these curves do not establish independent generalization.</figcaption></figure><dl class="run-facts"><dt>Completed updates in this phase</dt><dd>{steps}</dd><dt>Training rooms</dt><dd>{sampled_rooms} sampled / {rooms} configured</dd><dt>Target exposures in this phase</dt><dd>{exposures}</dd><dt>Initialization</dt><dd>Audited commercial-compatible lineage; parent provenance retained.</dd><dt>Camera conditioning</dt><dd>No camera inputs or trained camera head in this checkpoint. Image-grid RoPE is not a calibrated 3D camera encoding.</dd></dl></section>
<section class="limitations"><p class="eyebrow">04 / RESEARCH BOUNDARIES</p><h2>What this experiment does not establish.</h2><ul>{limits}</ul></section>
<section id="paper" class="paper"><p class="eyebrow">05 / TECHNICAL REPORT</p><h2>{title}</h2><p>The paper, figures and page are generated from the same checked experiment manifest. The PDF records architecture, dataset and training provenance, absolute metrics, visual examples and limitations. It is a technical draft, with no venue or peer-review claim.</p><div class="actions"><a class="button" href="paper.pdf">PDF ↗</a><a class="button secondary" href="paper.tex">LaTeX source ↓</a><a class="button secondary" href="results.json">Metrics & provenance ↓</a></div><p class="note"><a href="experiment.toml">Experiment TOML</a> · <a href="bundle.json">Bundle checksums</a> · <a href="media/samples.json">Sample selection</a> · <a href="media/latent-projection.json">Color projection</a></p></section>
</main><footer>burn_gekko · {author} · Generated from verified experiment artifacts; deployment is managed separately.<br>Procedural data: <a href="https://mosure.github.io/bevy_zeroverse/project/">bevy_zeroverse</a>.</footer><script id="sample-data" type="application/json">{data}</script><script src="explore.js"></script></body></html>"##,
        title = escape(&e.title),
        description = escape(&e.description),
        author = escape(&e.author),
        id = escape(&e.id),
        sample_mse = first.mse,
        sample_cos = first.cosine,
        short_hash = escape(&e.checkpoint_sha256[..16]),
        audit = escape(&serde_json::to_string_pretty(&metrics["input_audit"])?),
        architecture = escape(&e.architecture),
        steps = r.training["completed_steps"],
        rooms = r.training["config"]["train_rooms"],
        sampled_rooms = r.training["coverage"]["unique_rooms"],
        exposures = r.training["coverage"]["target_exposures"],
        task_heads = if e.output_heads.is_some() {
            "latent · RI · matching · RGB · calibration"
        } else if r.training["config"]["spatial_descriptor"].is_object() {
            "latent · RI · spatial matching"
        } else {
            "latent completion · RI"
        }
    );
    let html = html.replace(
        "<section id=\"explore\">",
        &format!("{output_heads}<section id=\"explore\">"),
    );
    let html = if e.output_heads.is_some() {
        html.replace(
            "<a href=\"#explore\">Explore</a>",
            "<a href=\"#output-heads\">RGB &amp; cameras</a><a href=\"#explore\">Latents</a>",
        )
    } else {
        html
    };
    let html = html.replace("No camera inputs or trained camera head in this checkpoint. Image-grid RoPE is not a calibrated 3D camera encoding.", camera_note);
    let html = html.replace(
        "<section id=\"evidence\">",
        &format!("{correspondence}<section id=\"evidence\">"),
    );
    let html = html.replace("</dl></section>", &format!("</dl>{efficiency}</section>"));
    let html = if let Some(note) = crate::display::encoder_stages(&r.training) {
        html.replace(
            "</dl>",
            &format!("</dl><p class='note'>{}</p>", escape(&note)),
        )
    } else {
        html
    };
    let html = if let Some(note) = crate::display::encoder_preservation(&r.training) {
        html.replace(
            "<section id=\"evidence\">",
            &format!(
                "<section id=\"evidence\"><p class='note'>{}</p>",
                escape(&note)
            ),
        )
    } else {
        html
    };
    let html = if let Some(note) = crate::display::view_geometry(&r.training) {
        html.replace(
            "<section id=\"evidence\">",
            &format!(
                "<section id=\"evidence\"><p class='note'>{}</p>",
                escape(&note)
            ),
        )
    } else {
        html
    };
    let html = if r.training["config"]["equivariance"].is_object()
        && r.training["scalar_windows"]["warp_pair_nll"].is_object()
    {
        let pair = &r.training["scalar_windows"]["warp_pair_nll"];
        let independent = &r.training["scalar_windows"]["warp_self_nll"];
        let note = format!(
            "<p class='note'>Training diagnostic: first versus last {}-update mean correspondence NLL was {:.3} → {:.3} for pair conditioning and {:.3} → {:.3} for same-image conditioning. These changing minibatches are descriptive convergence evidence; external transfer is evaluated separately.</p>",
            pair["window_updates"],
            crate::artifact::number(pair, "first_mean")?,
            crate::artifact::number(pair, "last_mean")?,
            crate::artifact::number(independent, "first_mean")?,
            crate::artifact::number(independent, "last_mean")?
        );
        html.replace(
            "<section id=\"evidence\">",
            &format!("<section id=\"evidence\">{note}"),
        )
    } else {
        html
    };
    let html = if pdf {
        html
    } else {
        html.replace("href=\"paper.pdf\"", "href=\"paper.tex\"")
            .replace("Read the paper ↗", "Read LaTeX source ↗")
            .replace("PDF ↗", "LaTeX ↗")
    };
    fs::write(out.join("index.html"), html)?;
    fs::write(out.join("style.css"), include_str!("assets/style.css"))?;
    fs::write(out.join("explore.js"), include_str!("assets/explore.js"))?;
    Ok(())
}

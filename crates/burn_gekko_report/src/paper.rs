use crate::{artifact::Report, experiment::Experiment, figures::Sample};
use anyhow::{Context, Result, ensure};
use burn_gekko_eval::schema::CapabilityStatus;
use std::{fs, path::Path, process::Command};
fn tex(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\\' => "\\textbackslash{}".into(),
            '&' => "\\&".into(),
            '%' => "\\%".into(),
            '$' => "\\$".into(),
            '#' => "\\#".into(),
            '_' => "\\_".into(),
            '{' => "\\{".into(),
            '}' => "\\}".into(),
            '~' => "\\textasciitilde{}".into(),
            '^' => "\\textasciicircum{}".into(),
            '↑' => "higher is better".into(),
            '↓' => "lower is better".into(),
            '·' => " / ".into(),
            '–' | '—' => "--".into(),
            _ => c.to_string(),
        })
        .collect()
}
pub fn write(
    e: &Experiment,
    r: &Report,
    samples: &[Sample],
    matches: &[crate::correspondence::CorrespondenceFigure],
    out: &Path,
    pdf: bool,
) -> Result<()> {
    let mut body = format!(
        r"\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=23mm]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{graphicx,booktabs,longtable,array,hyperref,microtype}}
\hypersetup{{hidelinks,pdftitle={{{}}},pdfauthor={{{}}}}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{6pt}}
\title{{{}}}\author{{{}}}\date{{Private technical draft / experiment {}}}
\begin{{document}}\maketitle
\begin{{abstract}}
{} This report presents a single selected training experiment in Burn, with fixed-teacher latent completion, co-visibility evaluation and declared correspondence readouts. Every table and visual binds to one checkpoint. Latent-space visualization is distinct from RGB synthesis; state-of-the-art performance is not established.
\end{{abstract}}
\section{{Model and task}}
{}

Sparse target tokens retain their original image-grid positions before encoder attention. Reference views are independently encoded. Fusion uses image-grid rotary position encodings and a shared reference role. No camera intrinsics, extrinsics or 3D rays enter this checkpoint. A fixed MIT V-JEPA 2.1 teacher supervises normalized latent targets; dense teacher features are used by the loss only. The full-target relative-improvement branch is evaluated separately from sparse completion.

\section{{Experiment and data}}
The selected checkpoint is \texttt{{{}}}. This phase completed {} updates over {} configured cached training rooms, with {} logged target exposures covering {} unique rooms. Configurations, weight ancestry, dataset identities and the selected checkpoint hash are retained in the bundled machine-readable results. User inputs are TOML; metrics and prediction artifacts are typed native Rust exports. Pretrained noncommercial Gekko weights are excluded from this training lineage.

Completion evaluation role: \textbf{{{}}}. Hidden-token MSE averages channels and hidden patches, then target views. Confidence intervals resample whole rooms rather than correlated pixels. The evaluation contains {} target views. Geometry is loaded after inference to score co-visibility and correspondence. Teacher-fitted display projections are visualization tools and do not enter inference or evaluation metrics.

\section{{Absolute results for the selected checkpoint}}
",
        tex(&e.title),
        tex(&e.author),
        tex(&e.title),
        tex(&e.author),
        tex(&e.id),
        tex(&e.description),
        tex(&e.architecture),
        &e.checkpoint_sha256[..16],
        r.training["completed_steps"],
        r.training["config"]["train_rooms"],
        r.training["coverage"]["target_exposures"],
        r.training["coverage"]["unique_rooms"],
        tex(&e.latent.evaluation_use),
        r.latent["target_views"]
    );
    for c in &r.capabilities {
        body.push_str(&format!(
            "\\subsection{{{}}}\n{}\n",
            tex(&c.label),
            tex(&c.protocol)
        ));
        if c.status == CapabilityStatus::Evaluated {
            body.push_str("\\begin{longtable}{p{.57\\linewidth}rp{.23\\linewidth}}\\toprule Metric & Value & Unit / count \\\\ \\midrule\n");
            for m in &c.metrics {
                let (value, unit) = crate::display::value(m);
                body.push_str(&format!(
                    "{} & {} & {} / {} \\\\\n",
                    tex(&m.label),
                    value,
                    tex(unit),
                    m.samples
                ));
            }
            body.push_str("\\bottomrule\\end{longtable}\n");
        } else {
            body.push_str("\\textbf{No evaluated capability is claimed for this head.}\n");
        }
        for l in &c.limitations {
            body.push_str(&format!("{}\n\n", tex(l)));
        }
    }
    let ci = &r.latent["room_bootstrap_mse"];
    body.push_str(&format!("Completion room-bootstrap 95\\% interval: [{:.6}, {:.6}], {} rooms, 10,000 deterministic replicates. This measures scene sampling uncertainty, not variation across training seeds.\n",ci["low"].as_f64().unwrap(),ci["high"].as_f64().unwrap(),ci["clusters"]));
    let gain = &r.latent["room_bootstrap_monocular_gain"];
    body.push_str(&format!("Paired gain from using references (monocular MSE minus cross-view MSE): {:.6}, room-bootstrap 95\\% interval [{:.6}, {:.6}]. Positive values favor reference use.\n", gain["mean"].as_f64().unwrap(),gain["low"].as_f64().unwrap(),gain["high"].as_f64().unwrap()));
    body.push_str("\\section{Visual evaluation protocol}\nExamples are evenly spaced over the exported sample identities, retaining the first and last. They are not selected by prediction quality. All displayed teacher features share one three-component PCA projection and teacher-derived 2nd--98th percentile color bounds; predictions use the identical projection and bounds. Color structure describes representation variation, not RGB texture reconstruction. Error maps use a fixed 0--1 scale (dark blue to teal to yellow to red), saturating above 1. Co-visibility truth uses teal for shared geometry, coral for non-visible patches and gray for observed/unknown patches.\n");
    for sample in samples {
        body.push_str(&format!(r"\clearpage\subsection{{Room {}, target view {}}}
\includegraphics[width=\linewidth]{{{}}}

Top row, left to right: sparse target input; reference 1; reference 2 (or an explicitly blank panel); full target RGB for evaluation only. Second row: teacher latent; predicted latent; hidden-token error; geometric co-visibility truth. If present, the third row shows the same model's monocular latent, learned RI score, signed reference benefit and geometric visibility fraction. RI/fraction maps use a 0--1 scale. Reference benefit is (monocular MSE minus cross-view MSE) divided by monocular MSE: -1 red (worse), 0 white, +1 teal (better), saturating outside that range. Unlike the clipped training RI target, this display retains negative effects. RI is an unconstrained regression score, not a calibrated visibility probability. The teacher and full target do not enter sparse inference.

Hidden-token MSE: {:.6}. Cosine: {:.6}. Values are recomputed in Rust from the exported arrays and checked against the population evaluation. All samples use the same error and projection conventions.
",sample.room_seed,sample.target_view,sample.contact_sheet,sample.mse,sample.cosine));
    }
    for pair in matches.chunks(2) {
        body.push_str("\\clearpage\\section*{Geometric correspondence examples}\n");
        for figure in pair {
            body.push_str(&format!(
                "\\subsection*{{{}}}\n\\includegraphics[width=.95\\linewidth]{{{}}}\n\n{}\n\n",
                tex(&figure.title),
                figure.file,
                tex(&figure.caption)
            ));
        }
    }
    if let Some(v) = &r.efficiency {
        body.push_str(&format!("\\section{{Training efficiency}}\nCommand duration {:.1} minutes; observed board energy {:.2} Wh; observed board joules per logged target {:.2}; telemetry coverage {:.1}\\%. {}\n",v.command_seconds/60.,v.observed_board_energy_wh,v.observed_board_joules_per_target,100.*v.telemetry_coverage,tex(&v.scope)));
        let batch = r.training["config"]["batch_size"]
            .as_f64()
            .context("training batch size")?;
        body.push_str("\\begin{center}\\begin{tabular}{lrrr}\\toprule Encoder stage & Median (ms) & p95 (ms) & Targets/s \\\\ \\midrule\n");
        for (stage, median) in &v.median_update_seconds_by_stage {
            let p95 = v.p95_update_seconds_by_stage[stage];
            body.push_str(&format!(
                "{} & {:.2} & {:.2} & {:.2} \\\\\n",
                tex(stage),
                median * 1000.,
                p95 * 1000.,
                batch / median
            ));
        }
        body.push_str("\\bottomrule\\end{tabular}\\end{center}\n");
        body.push_str(&format!(
            "Mean observed board power: {:.2} W. These are warmed per-update rates; complete-command energy retains preparation, validation and checkpoint overhead.\n",
            v.observed_mean_board_power_w
        ));
        if let Some(memory) = v.peak_process_vram_mib {
            body.push_str(&format!("Peak training-process VRAM: {memory:.0} MiB. "));
        }
        if let Some(memory) = v.peak_process_rss_mib {
            body.push_str(&format!("Peak training-process RSS: {memory:.0} MiB.\n"));
        }
    }
    if let Some(note) = crate::display::encoder_stages(&r.training) {
        body.push_str(&format!(
            "\n\\paragraph{{Encoder adaptation.}} {}\n",
            tex(&note)
        ));
    }
    if r.training["config"]["equivariance"].is_object()
        && r.training["scalar_windows"]["warp_pair_nll"].is_object()
    {
        let pair = &r.training["scalar_windows"]["warp_pair_nll"];
        let independent = &r.training["scalar_windows"]["warp_self_nll"];
        body.push_str(&format!("\\paragraph{{Training convergence diagnostic.}} First versus last {}-update mean correspondence NLL: pair conditioning {:.4} to {:.4}; same-image conditioning {:.4} to {:.4}. These are descriptive minibatch statistics, not an independent generalization estimate. Known-transform descriptor supervision is motivated by SiLK \\cite{{silk}}; implementation is independent, and no SiLK code or weights are used.\n",pair["window_updates"],crate::artifact::number(pair,"first_mean")?,crate::artifact::number(pair,"last_mean")?,crate::artifact::number(independent,"first_mean")?,crate::artifact::number(independent,"last_mean")?));
    }
    body.push_str(
        "\\clearpage\\section{Limitations and qualification requirements}\n\\begin{itemize}\n",
    );
    for l in &e.limitations {
        body.push_str(&format!("\\item {}\n", tex(l)));
    }
    body.push_str("\\end{itemize}\n");
    body.push_str(r"\section{Reproducibility and extension}
The model library, trainer, data reader, evaluation and publication are separate Rust crates. A new decoder/head supplies a checkpoint-bound capability record with explicit metric units, population, aggregation and evaluation status. Camera scoring includes SO(3) angular error, signed translation-direction error, normalized focal error and pose AUC at 5, 10 and 20 degrees. Zero ground-truth baselines are excluded from translation-direction and pose denominators; a zero predicted translation at a valid baseline is a failure. These implementations are not evidence that a camera head has been trained.

Generate this page and report from one TOML experiment manifest with \texttt{gekko-report build}. The bundle carries input and output hashes, exact sample identities, a common latent projection and the resolved report data. Local preparation does not commit, push, deploy or upload the repository. Checkpoints from old code identities require their sealed binaries for exact optimizer resume; weights-only new phases must record explicit ancestry and optimizer resets.

\section{Related work and provenance}
The project studies sparse visual encoding and multi-view fusion motivated by V-JEPA 2.1 and Gekko. Procedural room captures use published bevy\_zeroverse packages. Calibrated 3D positional encodings and supervised camera heads are research extensions; this experiment does not implement or claim their published results.
\begin{thebibliography}{9}
\bibitem{vjepa21} L. Mur-Labadia et al. V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning. \url{https://arxiv.org/abs/2603.14482v3}.
\bibitem{zeroverse} M. Mosure. bevy\_zeroverse: Procedural multi-view data. \url{https://mosure.github.io/bevy_zeroverse/project/}.
\bibitem{gekko} Revisiting Cross-View Completion: Self-Supervised Pre-Training via Reconstruction Error Comparison. \url{https://arxiv.org/abs/2609.01530}.
\bibitem{dppe} DPPE: Rethinking Camera-Based Positional Encoding for Scaling Multi-View Transformers. \url{https://arxiv.org/abs/2606.31585}.
\bibitem{zipsplat} ZipSplat: Fewer Gaussians, Better Splats. \url{https://arxiv.org/abs/2606.05102}.
\bibitem{silk} P. Gleize, W. Wang and M. Feiszli. SiLK: Simple Learned Keypoints. \url{https://arxiv.org/abs/2304.06194}.
\end{thebibliography}
\end{document}
");
    fs::write(out.join("paper.tex"), body)?;
    if pdf {
        for pass in 1..=2 {
            let result = Command::new("pdflatex")
                .current_dir(out)
                .args([
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-no-shell-escape",
                    "paper.tex",
                ])
                .output()
                .context("pdflatex is required for --pdf")?;
            fs::write(out.join(format!("paper-build-{pass}.log")), &result.stdout)?;
            ensure!(
                result.status.success(),
                "paper compilation failed; inspect paper-build-{pass}.log"
            );
        }
        ensure!(
            fs::read(out.join("paper.pdf"))?.starts_with(b"%PDF-"),
            "invalid compiled PDF"
        );
        for name in ["paper.aux", "paper.out", "paper.log"] {
            let _ = fs::remove_file(out.join(name));
        }
    }
    Ok(())
}

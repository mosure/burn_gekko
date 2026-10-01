//! A page and paper are two views of the same verified, single-run report data.
pub mod artifact;
mod correspondence;
mod display;
mod equivariance;
pub mod experiment;
mod figures;
mod latent_metrics;
mod learning;
mod output_heads;
mod page;
mod paper;
mod pose;
pub mod validate;
mod view_geometry;
use anyhow::{Result, ensure};
use std::{fs, path::Path};

pub fn build(spec: &Path, output: &Path, pdf: bool) -> Result<()> {
    ensure!(
        !output.exists(),
        "preserve existing publication bundle; use a fresh output directory"
    );
    let experiment: experiment::Experiment = burn_gekko_data::read_config(spec)?;
    let mut report = artifact::load(&experiment)?;
    fs::create_dir_all(output.join("media"))?;
    let samples = figures::build_samples(&experiment, output, &mut report.sources)?;
    let mut additional_sources = Vec::new();
    let mut correspondence = correspondence::build(&report, output, &mut additional_sources)?;
    correspondence.extend(equivariance::build(
        &experiment,
        output,
        &mut additional_sources,
    )?);
    correspondence.extend(view_geometry::build(
        &experiment,
        output,
        &mut additional_sources,
    )?);
    if let Some(pose) = &report.calibrated_pose {
        correspondence.extend(pose::build(pose, output, &mut additional_sources)?);
    }
    if let Some(heads) = &experiment.output_heads {
        correspondence.extend(output_heads::figures(
            heads,
            output,
            &mut additional_sources,
        )?);
    }
    report.sources.extend(additional_sources);
    learning::write(
        &experiment.run,
        report.training["starting_step"].as_u64().unwrap(),
        report.training["checkpoint_step"].as_u64().unwrap(),
        output,
        &mut report.sources,
    )?;
    page::write(&experiment, &report, &samples, &correspondence, output, pdf)?;
    paper::write(&experiment, &report, &samples, &correspondence, output, pdf)?;
    fs::copy(spec, output.join("experiment.toml"))?;
    burn_gekko_data::write_json(&output.join("results.json"), &report)?;
    let mut files = Vec::new();
    fn visit(root: &Path, dir: &Path, files: &mut Vec<serde_json::Value>) -> Result<()> {
        let mut entries = fs::read_dir(dir)?
            .map(|x| x.map(|x| x.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort();
        for p in entries {
            if p.is_dir() {
                visit(root, &p, files)?;
            } else {
                files.push(serde_json::json!({"file":p.strip_prefix(root)?.to_string_lossy(),"bytes":p.metadata()?.len(),"sha256":burn_gekko_data::sha256_file(&p)?}));
            }
        }
        Ok(())
    }
    visit(output, output, &mut files)?;
    burn_gekko_data::write_json(
        &output.join("bundle.json"),
        &serde_json::json!({"schema":1,"experiment_id":experiment.id,"checkpoint_sha256":experiment.checkpoint_sha256,"publication_status":"prepared research bundle; deployment is managed separately","files":files}),
    )?;
    Ok(())
}

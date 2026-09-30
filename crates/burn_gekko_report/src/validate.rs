//! Offline validation of a self-contained publication bundle.
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::Value;
use std::{
    fs,
    path::{Component, Path},
};
#[derive(Debug, Serialize)]
pub struct Validation {
    pub status: String,
    pub hashed_files: usize,
    pub decoded_images: usize,
    pub local_links: usize,
    pub pdf_present: bool,
    pub selected_samples: usize,
}
pub fn validate(root: &Path) -> Result<Validation> {
    let bundle: Value = serde_json::from_slice(&fs::read(root.join("bundle.json"))?)?;
    ensure!(bundle["schema"] == 1, "unsupported bundle schema");
    let entries = bundle["files"]
        .as_array()
        .context("bundle has no file list")?;
    let mut decoded_images = 0;
    for entry in entries {
        let name = entry["file"].as_str().context("bundle file name")?;
        let relative = Path::new(name);
        ensure!(
            !relative.is_absolute()
                && relative
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
            "invalid bundle path"
        );
        let p = root.join(relative);
        ensure!(
            p.metadata()?.len() == entry["bytes"].as_u64().context("bundle size")?
                && burn_gekko_data::sha256_file(&p)?
                    == entry["sha256"].as_str().context("bundle hash")?,
            "bundle file changed: {name}"
        );
        if name.ends_with(".png") {
            let image = image::open(&p)?;
            ensure!(image.width() > 0 && image.height() > 0, "empty visual");
            decoded_images += 1;
        }
    }
    let html = fs::read_to_string(root.join("index.html"))?;
    let mut local_links = 0;
    for (prefix, quote) in [
        ("href=\"", '"'),
        ("src=\"", '"'),
        ("href='", '\''),
        ("src='", '\''),
    ] {
        for part in html.split(prefix).skip(1) {
            let url = part.split(quote).next().context("unterminated link")?;
            if url.starts_with("https://") || url.starts_with("http://") {
                continue;
            }
            if let Some(anchor) = url.strip_prefix('#') {
                ensure!(
                    anchor.is_empty()
                        || html.contains(&format!("id=\"{anchor}\""))
                        || html.contains(&format!("id='{anchor}'")),
                    "missing anchor {anchor}"
                );
                continue;
            }
            ensure!(
                !url.is_empty() && !Path::new(url).is_absolute() && !url.contains(".."),
                "invalid local URL"
            );
            ensure!(root.join(url).is_file(), "missing local link {url}");
            local_links += 1;
        }
    }
    let samples: Vec<Value> = serde_json::from_slice(&fs::read(root.join("media/samples.json"))?)?;
    ensure!(!samples.is_empty(), "empty sample gallery");
    for sample in &samples {
        for panel in sample["panels"]
            .as_array()
            .context("missing sample panels")?
        {
            ensure!(
                root.join(panel[1].as_str().context("panel file")?)
                    .is_file(),
                "missing selectable image"
            );
        }
    }
    let pdf = root.join("paper.pdf").exists();
    if pdf {
        ensure!(
            fs::read(root.join("paper.pdf"))?.starts_with(b"%PDF-"),
            "invalid PDF header"
        );
    }
    Ok(Validation {
        status:
            "native_hash_image_link_validation_passed; browser interactions require separate check"
                .into(),
        hashed_files: entries.len(),
        decoded_images,
        local_links,
        pdf_present: pdf,
        selected_samples: samples.len(),
    })
}

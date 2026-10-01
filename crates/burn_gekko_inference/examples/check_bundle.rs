//! Explicit CPU numerical reference for a packaged model and two to four PNG/JPEG views.
use anyhow::{Result, ensure};
use burn_gekko_inference::{Bundle, Inference, Request, RgbInput};
use std::{fs, path::Path};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        (5..=7).contains(&args.len()),
        "check_bundle model-directory output.json image1 image2 [image3 image4]"
    );
    let root = Path::new(&args[1]);
    let bundle = Bundle::parse(&fs::read_to_string(root.join("manifest.toml"))?)?;
    let mut foundation = Vec::new();
    for part in &bundle.foundation {
        let bytes = fs::read(root.join(&part.name))?;
        part.verify(&bytes)?;
        foundation.extend(bytes);
    }
    let camera = fs::read(root.join(&bundle.camera.name))?;
    let rgb = fs::read(root.join(&bundle.rgb.name))?;
    let images = args[3..]
        .iter()
        .map(|p| RgbInput::decode(p.clone(), &fs::read(p)?, bundle.image_size))
        .collect::<Result<Vec<_>>>()?;
    let model = Inference::<burn::backend::NdArray<f32>>::load(
        bundle,
        foundation,
        camera,
        rgb,
        Default::default(),
    )?;
    let started = std::time::Instant::now();
    let result = pollster::block_on(model.run(Request {
        revision: 0,
        target: 0,
        images,
    }))?;
    fs::write(
        &args[2],
        serde_json::to_vec_pretty(
            &serde_json::json!({"result": result, "inference_seconds": started.elapsed().as_secs_f64()}),
        )?,
    )?;
    println!("CPU reference written to {}", args[2]);
    Ok(())
}

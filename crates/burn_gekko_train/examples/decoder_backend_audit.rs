//! Compare decoder predictions and input gradients across CPU and CUDA.
//! This checks backend numerics, not parity with the published Gekko architecture.
use anyhow::{Result, ensure};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, TensorData, backend::AutodiffBackend},
};
use burn_gekko_train::{
    loss::reconstruction_loss,
    model::{DecoderConfig, DecoderPosition, GekkoDecoder},
};
use burn_vjepa::SparseTokenMask;
use std::{collections::BTreeMap, fs, path::Path};

fn values(count: usize, phase: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (i as f32 * 0.017 + phase).sin() * 0.4 + 0.2)
        .collect()
}

fn tensor<B: AutodiffBackend>(shape: [usize; 3], phase: f32) -> Tensor<B, 3> {
    Tensor::from_data(
        TensorData::new(values(shape.iter().product(), phase), shape),
        &Default::default(),
    )
    .require_grad()
}

fn run<B: AutodiffBackend>(
    directory: &Path,
    backend: &str,
    create: bool,
) -> Result<BTreeMap<String, Vec<f32>>> {
    B::seed(&Default::default(), 29);
    let mut model = GekkoDecoder::<B>::with_reconstruction(
        &DecoderConfig {
            encoder_dim: 1536,
            width: 384,
            depth: 2,
            heads: 6,
            patch: 16,
        },
        DecoderPosition::Rope2d,
        true,
        &Default::default(),
    )?
    .with_mae_context_before_self(true);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    if create {
        model
            .clone()
            .save_file(directory.join("decoder"), &recorder)?;
    }
    // A lazy Param clone initializes independently. Both backends must reload
    // this same file, including the CPU model that created it.
    model = model.load_file(directory.join("decoder"), &recorder, &Default::default())?;
    let masked = tensor::<B>([2, 4, 1536], 0.1);
    let full = tensor::<B>([2, 16, 1536], 0.3);
    let reference_a = tensor::<B>([2, 16, 1536], 0.6);
    let reference_b = tensor::<B>([2, 16, 1536], 1.2);
    let mask = SparseTokenMask::new(vec![0, 5, 9, 14], 16)?;
    let predictions = model.forward(
        masked.clone(),
        full.clone(),
        vec![reference_a.clone(), reference_b.clone()],
        &mask,
        [4, 4],
    )?;
    let mut result = BTreeMap::new();
    for (name, weight) in [
        ("cross_head_weight", model.cross_rgb.weight.val()),
        ("mae_head_weight", model.mae_rgb.weight.val()),
        ("ri_head_weight", model.ri.weight.val()),
    ] {
        result.insert(name.into(), weight.into_data().to_vec::<f32>()?);
    }
    for (name, value) in [
        ("cross", predictions.cross_rgb.clone()),
        ("mae", predictions.mae_rgb.clone()),
        ("ri", predictions.ri.clone()),
    ] {
        result.insert(name.into(), value.into_data().to_vec::<f32>()?);
    }
    let target = tensor::<B>([2, 16, 768], 0.8).detach();
    let losses = reconstruction_loss(predictions, target, &mask, true, true);
    result.insert(
        "loss".into(),
        losses.total.clone().into_data().to_vec::<f32>()?,
    );
    let grads = losses.total.backward();
    for (name, value) in [
        ("masked_gradient", masked),
        ("full_gradient", full),
        ("reference_a_gradient", reference_a),
        ("reference_b_gradient", reference_b),
    ] {
        let gradient = value.grad(&grads).expect("input gradient required");
        result.insert(name.into(), gradient.into_data().to_vec::<f32>()?);
    }
    burn_gekko_data::write_json(&directory.join(format!("{backend}.json")), &result)?;
    Ok(result)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 2,
        "usage: decoder_backend_audit .data/NEW_DIRECTORY"
    );
    let directory = Path::new(&args[1]);
    ensure!(
        directory.starts_with(".data") && !directory.exists(),
        "use a new .data directory"
    );
    fs::create_dir_all(directory)?;
    let cpu = run::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(directory, "cpu", true)?;
    #[cfg(feature = "cuda")]
    {
        let cuda = run::<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>(
            directory, "cuda", false,
        )?;
        let mut metrics = BTreeMap::new();
        let mut passed = true;
        for (name, reference) in cpu {
            let candidate = &cuda[&name];
            let energy: f64 = reference.iter().map(|&x| (x as f64).powi(2)).sum();
            let error: f64 = reference
                .iter()
                .zip(candidate)
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum();
            let relative_rms = (error / energy.max(1e-30)).sqrt();
            let max_abs = reference
                .iter()
                .zip(candidate)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            passed &= relative_rms.is_finite() && relative_rms < 0.01;
            metrics.insert(
                name,
                serde_json::json!({"relative_rms":relative_rms,"max_abs":max_abs}),
            );
        }
        burn_gekko_data::write_json(
            &directory.join("comparison.json"),
            &serde_json::json!({"passed":passed,"relative_rms_gate":0.01,"scope":"two-layer width-384 decoder, batch two, 4x4 token grid, normalized content plus learned calibration, corrected MAE context; CPU/CUDA predictions and input gradients; not official Gekko parity","metrics":metrics}),
        )?;
        ensure!(passed, "CPU/CUDA decoder numerical gate failed");
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = cpu;
        anyhow::bail!("CUDA feature required for the paired check");
    }
}

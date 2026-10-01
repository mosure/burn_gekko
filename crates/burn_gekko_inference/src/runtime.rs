use crate::{Bundle, RgbInput, bundle::digest, pixels::unpatch};
use anyhow::{Result, ensure};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::{Tensor, backend::Backend},
};
use burn_gekko::{
    encoder::{upload_rgb, visible_mask},
    heads::{calibration::CalibrationHead, reconstruction::RgbReconstructionHead},
    latent::LatentModel,
};
use burn_gekko_metrics::heads::{RgbScore, rgb_score, rotation_from_six};
use burn_vjepa::VJepaEncoder;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Request {
    pub revision: u64,
    pub target: usize,
    /// First reference is also the anchor frame of the predicted camera.
    pub images: Vec<RgbInput>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Match {
    pub target: [f32; 2],
    pub reference: [f32; 2],
    pub cosine: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub revision: u64,
    pub target: usize,
    pub reference: usize,
    pub size: usize,
    pub visible: Vec<usize>,
    pub rgb: Vec<u8>,
    pub monocular_rgb: Vec<u8>,
    pub rgb_score: RgbScore,
    pub monocular_score: RgbScore,
    /// Full-target RI branch, an unconstrained regression score, not a probability.
    pub improvement: Vec<f32>,
    pub camera: Vec<f32>,
    pub rotation: Option<[[f64; 3]; 3]>,
    pub focal: [f64; 2],
    pub matches: Vec<Match>,
}

pub struct Inference<B: Backend> {
    pub bundle: Bundle,
    model: LatentModel<B>,
    camera: CalibrationHead<B>,
    rgb: RgbReconstructionHead<B>,
    device: B::Device,
}
impl<B: Backend> Inference<B> {
    pub fn load(
        bundle: Bundle,
        foundation: Vec<u8>,
        camera: Vec<u8>,
        rgb: Vec<u8>,
        device: B::Device,
    ) -> Result<Self> {
        bundle.validate()?;
        ensure!(
            digest(&foundation) == bundle.foundation_sha256
                && digest(&camera) == bundle.camera_sha256
                && digest(&rgb) == bundle.rgb_sha256,
            "weight identity mismatch"
        );
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let model = LatentModel::new(
            VJepaEncoder::new(&bundle.encoder, &device),
            bundle.encoder.clone(),
            &bundle.decoder,
            &device,
        )?
        .prepare_spatial_descriptor(Some(&bundle.spatial_descriptor))?
        .load_checked_record(recorder.load(foundation, &device)?)?
        .with_spatial_input_layer(Some(bundle.spatial_input_layer), false)?;
        let mut model = model;
        model.fusion.decoder = model
            .fusion
            .decoder
            .with_cross_view_rope(true)
            .with_stable_attention(false);
        model.validate_spatial_descriptor()?;
        let camera = CalibrationHead::new(bundle.decoder.width, bundle.head_width, &device)
            .load_record(recorder.load(camera, &device)?);
        let rgb = RgbReconstructionHead::new(
            bundle.encoder.encoder.embed_dim,
            bundle.head_width,
            &device,
        )
        .load_record(recorder.load(rgb, &device)?);
        Ok(Self {
            bundle,
            model,
            camera,
            rgb,
            device,
        })
    }
    pub async fn run(&self, request: Request) -> Result<InferenceOutput> {
        let size = self.bundle.image_size;
        ensure!(
            (2..=4).contains(&request.images.len()) && request.target < request.images.len(),
            "provide two to four views"
        );
        for image in &request.images {
            image.validate()?;
            ensure!(image.size == size, "image size differs from model");
        }
        let rgb: Vec<_> = request
            .images
            .iter()
            .map(|i| {
                upload_rgb::<B>(
                    i.pixels.iter().map(|v| *v as f32 / 255.).collect(),
                    [1, size, size],
                    &self.device,
                )
            })
            .collect();
        let grid = [size / 16, size / 16];
        let n = grid[0] * grid[1];
        let mask = visible_mask(n, self.bundle.mask_ratio, self.bundle.mask_seed, 0)?;
        let refs: Vec<_> = (1..request.images.len())
            .map(|offset| (request.target + offset) % request.images.len())
            .collect();
        let reference = refs[0];
        let encoded = self.model.encode_references(&rgb);
        let predicted = self.model.predict_encoded(
            self.model.encode(rgb[request.target].clone(), Some(&mask)),
            refs.iter().map(|i| encoded[*i].clone()).collect(),
            &mask,
            grid,
        )?;
        // Read completion before dense annotation routes; neither uses dense target features.
        let pixels = values(self.rgb.forward(predicted.cross)).await?;
        let mono = values(self.rgb.forward(predicted.monocular)).await?;
        let a = self.model.fusion.decoder.pair_features(
            encoded[request.target].clone(),
            encoded[reference].clone(),
            grid,
        )?;
        let b = self.model.fusion.decoder.pair_features(
            encoded[reference].clone(),
            encoded[request.target].clone(),
            grid,
        )?;
        let da = self
            .model
            .spatial_descriptor(encoded[request.target].clone(), a.clone())
            .ok_or_else(|| anyhow::anyhow!("missing spatial head"))?;
        let db = self
            .model
            .spatial_descriptor(encoded[reference].clone(), b.clone())
            .ok_or_else(|| anyhow::anyhow!("missing spatial head"))?;
        let normalize =
            |x: Tensor<B, 3>| x.clone() / x.powf_scalar(2.).sum_dim(2).clamp_min(1e-12).sqrt();
        let similarity = values(normalize(da).matmul(normalize(db).swap_dims(1, 2))).await?;
        let camera = values(
            self.camera
                .forward(Tensor::cat(vec![pool(a, grid), pool(b, grid)], 1)),
        )
        .await?;
        let improvement = values(self.model.predict_improvement(
            encoded[request.target].clone(),
            refs.iter().map(|i| encoded[*i].clone()).collect(),
            grid,
        )?)
        .await?;
        let hidden: Vec<_> = (0..n).filter(|i| !mask.indices().contains(i)).collect();
        let truth = request.images[request.target].patches();
        Ok(InferenceOutput {
            revision: request.revision,
            target: request.target,
            reference,
            size,
            visible: mask.indices().to_vec(),
            rgb: unpatch(&pixels, size)?,
            monocular_rgb: unpatch(&mono, size)?,
            rgb_score: rgb_score(&pixels, &truth, &hidden)?,
            monocular_score: rgb_score(&mono, &truth, &hidden)?,
            rotation: rotation_from_six(&camera[..6]),
            focal: [
                (camera[9] as f64).clamp(-4., 4.).exp(),
                (camera[10] as f64).clamp(-4., 4.).exp(),
            ],
            camera,
            improvement,
            matches: mutual_matches(&similarity, grid),
        })
    }
}
async fn values<B: Backend, const D: usize>(x: Tensor<B, D>) -> Result<Vec<f32>> {
    let out = x
        .into_data_async()
        .await?
        .convert::<f32>()
        .to_vec::<f32>()?;
    ensure!(
        out.iter().all(|x| x.is_finite()),
        "nonfinite inference result"
    );
    Ok(out)
}
fn pool<B: Backend>(x: Tensor<B, 3>, grid: [usize; 2]) -> Tensor<B, 3> {
    let [b, _, d] = x.dims();
    x.reshape([b, 4, grid[0] / 4, 4, grid[1] / 4, d])
        .mean_dim(2)
        .mean_dim(4)
        .reshape([b, 16, d])
}
fn mutual_matches(similarity: &[f32], grid: [usize; 2]) -> Vec<Match> {
    let n = grid[0] * grid[1];
    let best: Vec<_> = (0..n)
        .map(|i| {
            (0..n)
                .max_by(|a, b| similarity[i * n + *a].total_cmp(&similarity[i * n + *b]))
                .unwrap()
        })
        .collect();
    let reverse: Vec<_> = (0..n)
        .map(|j| {
            (0..n)
                .max_by(|a, b| similarity[*a * n + j].total_cmp(&similarity[*b * n + j]))
                .unwrap()
        })
        .collect();
    let xy = |i: usize| {
        [
            ((i % grid[1]) * 16 + 8) as f32,
            ((i / grid[1]) * 16 + 8) as f32,
        ]
    };
    let mut matches: Vec<_> = (0..n)
        .filter(|i| reverse[best[*i]] == *i)
        .map(|i| Match {
            target: xy(i),
            reference: xy(best[i]),
            cosine: similarity[i * n + best[i]],
        })
        .collect();
    matches.sort_by(|a, b| b.cosine.total_cmp(&a.cosine));
    matches.truncate(24);
    matches
}

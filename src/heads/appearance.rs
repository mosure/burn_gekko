//! Optional RGB-only appearance transport beside the generative RGB head.
//! This head receives decoder features and reference RGB, never hidden target RGB.
use crate::transport::{TransportOutput, pixel_grid, resize, sample_pixels};
use burn::{
    module::Module,
    nn::{Initializer, Linear, LinearConfig},
    tensor::{Tensor, activation, backend::Backend},
};

#[derive(Module, Debug)]
pub struct AppearanceHead<B: Backend> {
    pub flow: Linear<B>,
    pub matcher: Option<crate::matching::RgbMatcher<B>>,
    #[module(skip)]
    pub max_displacement: f32,
    #[module(skip)]
    pub auxiliary_weight: f64,
    #[module(skip)]
    pub pyramid_loss: bool,
    #[module(skip)]
    pub coarse_smoothness_weight: f64,
}

pub struct AppearanceOutput<B: Backend> {
    pub rgb: Tensor<B, 4>,
    pub transport: TransportOutput<B>,
    pub mixture: Tensor<B, 4>,
}

impl<B: Backend> AppearanceHead<B> {
    pub fn new(
        width: usize,
        max_displacement: f32,
        auxiliary_weight: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            // Four subpatch samples per axis: dx, dy, reference confidence,
            // and generative confidence. Shared parameters preserve set symmetry.
            flow: LinearConfig::new(width, 4 * 4 * 4)
                .with_initializer(Initializer::Normal {
                    mean: 0.,
                    std: 0.0001,
                })
                .init(device),
            matcher: None,
            max_displacement,
            auxiliary_weight,
            pyramid_loss: false,
            coarse_smoothness_weight: 0.,
        }
    }

    pub fn with_pyramid_loss(mut self, enabled: bool) -> Self {
        self.pyramid_loss = enabled;
        self
    }

    pub fn with_coarse_smoothness(mut self, weight: f64) -> Self {
        self.coarse_smoothness_weight = weight;
        self
    }

    pub fn forward(
        &self,
        features: Vec<Tensor<B, 3>>,
        references: &[Tensor<B, 4>],
        generated: Tensor<B, 4>,
        query_rgb: Tensor<B, 4>,
        grid: [usize; 2],
    ) -> AppearanceOutput<B> {
        assert!(!features.is_empty() && features.len() == references.len());
        let [b, _, h, w] = generated.dims();
        let (x, y) = pixel_grid(b, h, w, &generated.device());
        let mut warped = Vec::new();
        let mut flows = Vec::new();
        let mut coarse_flows = Vec::new();
        let mut reference_logits = Vec::new();
        let mut generated_logits = Vec::new();
        let matching_query = self.matcher.as_ref().map(|m| m.encode(query_rgb));
        for (feature, rgb) in features.into_iter().zip(references) {
            let coarse = self
                .flow
                .forward(feature)
                .reshape([b, grid[0], grid[1], 4, 4, 4])
                .permute([0, 5, 1, 3, 2, 4])
                .reshape([b, 4, grid[0] * 4, grid[1] * 4]);
            if self.coarse_smoothness_weight > 0. && self.matcher.is_none() {
                coarse_flows
                    .push(coarse.clone().slice_dim(1, 0..2).tanh() * (self.max_displacement / 4.));
            }
            let dense = resize(coarse, h, w);
            let mut flow = dense.clone().slice_dim(1, 0..2).tanh() * self.max_displacement;
            if let Some(matcher) = &self.matcher {
                flow = crate::matching::refine_flow(
                    matching_query.as_ref().unwrap().clone(),
                    matcher.encode(rgb.clone()),
                    flow,
                    self.max_displacement,
                );
                if self.coarse_smoothness_weight > 0. {
                    coarse_flows.push(
                        burn::tensor::module::avg_pool2d(
                            flow.clone(),
                            [4, 4],
                            [4, 4],
                            [0, 0],
                            false,
                            false,
                        ) / 4.,
                    );
                }
            }
            warped.push(sample_pixels(
                rgb.clone(),
                x.clone() + flow.clone().slice_dim(1, 0..1),
                y.clone() + flow.clone().slice_dim(1, 1..2),
            ));
            flows.push(flow);
            reference_logits.push(dense.clone().slice_dim(1, 2..3));
            generated_logits.push(dense.slice_dim(1, 3..4));
        }
        let reference_logits = Tensor::cat(reference_logits, 1);
        let weights = activation::softmax(reference_logits.clone(), 1);
        let combine = |weights: Tensor<B, 4>| {
            let terms: Vec<_> = warped
                .iter()
                .enumerate()
                .map(|(i, rgb)| rgb.clone() * weights.clone().slice_dim(1, i..i + 1))
                .collect();
            Tensor::stack::<5>(terms, 0).sum_dim(0).squeeze_dim(0)
        };
        let transported = combine(weights.clone());
        let generated_logit = Tensor::cat(generated_logits, 1).mean_dim(1);
        // Begin close to the existing generator; an auxiliary photometric loss
        // trains transport even when the mixture initially prefers generation.
        let mixture = activation::softmax(
            Tensor::cat(vec![generated_logit, reference_logits - 2.], 1),
            1,
        );
        let rgb = generated * mixture.clone().slice_dim(1, 0..1)
            + combine(mixture.clone().slice_dim(1, 1..references.len() + 1));
        AppearanceOutput {
            rgb,
            transport: TransportOutput {
                rgb: transported,
                warped,
                flows,
                coarse_flows,
                weights,
            },
            mixture,
        }
    }
}

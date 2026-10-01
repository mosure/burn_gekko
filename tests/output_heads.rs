use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::{Tensor, TensorData, backend::Backend},
};
use burn_gekko::{
    heads::{
        calibration::{CalibrationHead, calibration_loss},
        reconstruction::RgbReconstructionHead,
    },
    tensor::{scalar, values},
};
type B = Autodiff<NdArray<f32>>;

#[test]
fn camera_and_rgb_heads_fit_without_nonfinite_gradients_across_initializations() {
    let device = Default::default();
    for seed in [17, 29, 41] {
        B::seed(&device, seed);
        let mut camera = CalibrationHead::<B>::new(8, 16, &device);
        let mut rgb = RgbReconstructionHead::<B>::new(8, 16, &device);
        let mut camera_opt = AdamWConfig::new().with_weight_decay(0.).init();
        let mut rgb_opt = AdamWConfig::new().with_weight_decay(0.).init();
        let x = Tensor::<B, 3>::from_data(
            TensorData::new(
                (0..2 * 32 * 8).map(|i| (i as f32 * 0.019).sin()).collect(),
                [2, 32, 8],
            ),
            &device,
        );
        let y = Tensor::<B, 2>::from_data(
            TensorData::new(
                vec![
                    1., 0., 0., 0., 1., 0., 0.6, 0.8, 0., -0.4, -0.4, 0., 1., 0., -1., 0., 0.,
                    -0.8, 0.6, 0., -0.2, -0.2,
                ],
                [2, 11],
            ),
            &device,
        );
        let z = x.clone().slice_dim(1, 0..2);
        let target = Tensor::<B, 3>::from_data(
            TensorData::new(
                (0..2 * 2 * 768)
                    .map(|i| 0.3 + 0.15 * (i as f32 * 0.013).sin())
                    .collect(),
                [2, 2, 768],
            ),
            &device,
        );
        let loss = |head: &CalibrationHead<B>| {
            calibration_loss(
                head.forward(x.clone()),
                y.clone(),
                Tensor::ones([2, 1], &device),
            )
        };
        let first = scalar(loss(&camera)).unwrap();
        let first_rgb = scalar(
            (rgb.forward(z.clone()) - target.clone())
                .powf_scalar(2.)
                .mean(),
        )
        .unwrap();
        for _ in 0..500 {
            let l = loss(&camera);
            assert!(scalar(l.clone()).unwrap().is_finite());
            let g = GradientsParams::from_grads(l.backward(), &camera);
            camera = camera_opt.step(0.002, camera, g);
            let l = (rgb.forward(z.clone()) - target.clone())
                .powf_scalar(2.)
                .mean();
            let g = GradientsParams::from_grads(l.backward(), &rgb);
            rgb = rgb_opt.step(0.002, rgb, g);
        }
        let last = scalar(loss(&camera)).unwrap();
        let last_rgb = scalar(
            (rgb.forward(z.clone()) - target.clone())
                .powf_scalar(2.)
                .mean(),
        )
        .unwrap();
        eprintln!("seed {seed}: camera {first} -> {last}; RGB {first_rgb} -> {last_rgb}");
        assert!(last < first * 0.25);
        assert!(last_rgb < first_rgb * 0.25);
        let prediction = values(rgb.valid().forward(z.clone().inner())).unwrap();
        assert!(prediction.iter().all(|v| (0. ..=1.).contains(v)));
        // Once the zero-initialized output has warmed up, gradients reach the input trunk.
        let input = x.clone().detach().require_grad();
        let grad = calibration_loss(
            camera.forward(input.clone()),
            y.clone(),
            Tensor::ones([2, 1], &device),
        )
        .backward();
        assert!(scalar(input.grad(&grad).unwrap().abs().max()).unwrap() > 0.);
        let input = z.clone().detach().require_grad();
        let grad = (rgb.forward(input.clone()) - target.clone())
            .powf_scalar(2.)
            .mean()
            .backward();
        assert!(scalar(input.grad(&grad).unwrap().abs().max()).unwrap() > 0.);
    }
}

#[test]
fn zero_baselines_have_no_translation_gradient_and_labels_are_detached() {
    let device = Default::default();
    let x = Tensor::<B, 2>::zeros([1, 11], &device).require_grad();
    let y = Tensor::<B, 2>::ones([1, 11], &device).require_grad();
    let grad = calibration_loss(x.clone(), y.clone(), Tensor::zeros([1, 1], &device)).backward();
    assert!(y.grad(&grad).is_none());
    assert_eq!(
        scalar(x.grad(&grad).unwrap().slice_dim(1, 6..9).abs().max()).unwrap(),
        0.
    );
}

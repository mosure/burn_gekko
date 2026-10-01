//! Renderer-supervised correspondences, kept separate from RGB model inputs.
//! Compact immutable CPU caches avoid decoding geometry in the training loop.
use crate::{GeometryScene, project};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
mod cache;
pub use cache::{Cache, Manifest, PrepareConfig, prepare};

pub const POLICY: &str = "v1: mean world position of the central 2x2 source pixels; all depths positive and spread <= max(0.02m,1% mean depth); source reprojection within 0.1px; nearest reference depth agreement <= max(0.02m,1% projected depth); bilinear descriptor labels only inside patch-center hull; unknown/occluded/out-of-view excluded";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum State {
    Unknown,
    Visible,
    Occluded,
    OutOfView,
}

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub state: State,
    pub xy: [f32; 2],
}

/// Continuous patch-center labels. Surface discontinuities are unknown, never negatives.
pub fn pair(
    scene: &GeometryScene,
    target: usize,
    reference: usize,
    patch: usize,
) -> Result<Vec<Point>> {
    let (w, h) = (scene.width, scene.height);
    ensure!(
        patch >= 2
            && patch.is_multiple_of(2)
            && w.is_multiple_of(patch)
            && h.is_multiple_of(patch)
            && target < scene.depth.len()
            && reference < scene.depth.len(),
        "invalid geometry target grid/views"
    );
    let mut points = Vec::with_capacity(w / patch * (h / patch));
    for y in 0..h / patch {
        for x in 0..w / patch {
            let center = [
                (x * patch + patch / 2) as f32,
                (y * patch + patch / 2) as f32,
            ];
            let mut point = Point {
                state: State::Unknown,
                xy: [0.; 2],
            };
            let (cx, cy) = (center[0] as usize, center[1] as usize);
            let pixels =
                [(cx - 1, cy - 1), (cx, cy - 1), (cx - 1, cy), (cx, cy)].map(|(a, b)| b * w + a);
            let depths = pixels.map(|i| scene.depth[target][i]);
            let mean = depths.iter().sum::<f32>() * 0.25;
            let spread = depths.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - depths.iter().copied().fold(f32::INFINITY, f32::min);
            if depths.iter().all(|d| d.is_finite() && *d > 0.)
                && spread <= 0.02_f32.max(0.01 * mean)
            {
                let world = std::array::from_fn(|c| {
                    pixels
                        .iter()
                        .map(|i| scene.position[target][i * 3 + c])
                        .sum::<f32>()
                        * 0.25
                });
                let self_projected = project(
                    world,
                    &scene.world_from_view[target],
                    scene.fovy[target],
                    w,
                    h,
                );
                if world.iter().all(|v| v.is_finite())
                    && self_projected
                        .is_some_and(|p| (p[0] - center[0]).hypot(p[1] - center[1]) <= 0.1)
                {
                    point.state = State::OutOfView;
                    if let Some([u, v, z]) = project(
                        world,
                        &scene.world_from_view[reference],
                        scene.fovy[reference],
                        w,
                        h,
                    ) && u.is_finite()
                        && v.is_finite()
                        && u >= 0.
                        && v >= 0.
                        && u < w as f32
                        && v < h as f32
                    {
                        point.xy = [u, v];
                        let depth =
                            scene.depth[reference][v.floor() as usize * w + u.floor() as usize];
                        let tolerance = 0.02_f32.max(0.01 * z);
                        point.state = if !depth.is_finite() || depth <= 0. {
                            State::Unknown
                        } else if (z - depth).abs() <= tolerance {
                            State::Visible
                        } else if z > depth + tolerance {
                            State::Occluded
                        } else {
                            State::Unknown
                        };
                    }
                }
            }
            points.push(point);
        }
    }
    Ok(points)
}

/// Zero rows are excluded by the correspondence objective; border locations are not clamped.
pub fn labels(points: &[Point], grid: [usize; 2], patch: usize) -> Result<(Vec<f32>, usize)> {
    let [h, w] = grid;
    let n = h * w;
    ensure!(
        n > 0 && patch > 0 && points.len() == n,
        "invalid correspondence rows"
    );
    let mut out = vec![0.; n * n];
    let mut valid = 0;
    for (query, point) in points.iter().enumerate() {
        if point.state != State::Visible {
            continue;
        }
        let [u, v] = point.xy.map(|x| x / patch as f32 - 0.5);
        ensure!(u.is_finite() && v.is_finite(), "nonfinite visible location");
        if u < 0. || v < 0. || u > (w - 1) as f32 || v > (h - 1) as f32 {
            continue;
        }
        let (a, b) = (u.floor() as usize, v.floor() as usize);
        let (dx, dy) = (u - a as f32, v - b as f32);
        for (x, y, p) in [
            (a, b, (1. - dx) * (1. - dy)),
            ((a + 1).min(w - 1), b, dx * (1. - dy)),
            (a, (b + 1).min(h - 1), (1. - dx) * dy),
            ((a + 1).min(w - 1), (b + 1).min(h - 1), dx * dy),
        ] {
            out[query * n + y * w + x] += p;
        }
        valid += 1;
    }
    Ok((out, valid))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plane() -> GeometryScene {
        let mut positions = Vec::new();
        for y in 0..32 {
            for x in 0..48 {
                positions.extend([
                    (x as f32 + 0.5 - 24.) / 16.,
                    -(y as f32 + 0.5 - 16.) / 16.,
                    -2.,
                ]);
            }
        }
        GeometryScene {
            width: 48,
            height: 32,
            depth: vec![vec![2.; 48 * 32]; 2],
            position: vec![positions; 2],
            world_from_view: vec![
                [
                    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.
                ];
                2
            ],
            fovy: vec![2. * 0.5_f32.atan(); 2],
        }
    }
    #[test]
    fn continuous_centers_translation_occlusion_and_unknown_are_distinct() {
        let mut scene = plane();
        let (identity, n) = labels(&pair(&scene, 0, 1, 16).unwrap(), [2, 3], 16).unwrap();
        assert_eq!(n, 6);
        for i in 0..6 {
            assert!((identity[i * 6 + i] - 1.).abs() < 1e-6);
        }
        scene.world_from_view[1][12] = 1.; // one patch left at z=2, fx=32
        let points = pair(&scene, 0, 1, 16).unwrap();
        assert_eq!(points[0].state, State::OutOfView);
        let (shift, n) = labels(&points, [2, 3], 16).unwrap();
        assert_eq!(n, 4);
        assert!((shift[6] - 1.).abs() < 1e-6);
        scene.depth[1].fill(1.);
        assert_eq!(pair(&scene, 0, 1, 16).unwrap()[1].state, State::Occluded);
        scene.depth[1].fill(3.);
        assert_eq!(pair(&scene, 0, 1, 16).unwrap()[1].state, State::Unknown);
        scene.depth[1].fill(2.);
        scene.depth[0][7 * 48 + 23] = 0.;
        assert_eq!(pair(&scene, 0, 1, 16).unwrap()[1].state, State::Unknown);
    }
    #[test]
    fn border_visible_and_nonvisible_rows_are_not_supervised() {
        let points = [
            Point {
                state: State::Visible,
                xy: [12., 8.],
            },
            Point {
                state: State::Visible,
                xy: [31., 8.],
            },
            Point {
                state: State::Occluded,
                xy: [8., 24.],
            },
            Point {
                state: State::Unknown,
                xy: [24., 24.],
            },
        ];
        let (p, n) = labels(&points, [2, 2], 16).unwrap();
        assert_eq!(n, 1);
        assert_eq!(&p[..4], &[0.75, 0.25, 0., 0.]);
        assert!(p[4..].iter().all(|x| *x == 0.));
    }
}

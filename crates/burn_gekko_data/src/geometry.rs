use crate::{GeometryScene, Split, load_geometry, open_dataset};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::path::Path;

/// Distinguish missing reference depth from a known invisible point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    Visible,
    Occluded,
    OutOfView,
    Unknown,
}

pub fn project(
    world: [f32; 3],
    camera: &[f32; 16],
    fovy: f32,
    width: usize,
    height: usize,
) -> Option<[f32; 3]> {
    let p = [
        world[0] - camera[12],
        world[1] - camera[13],
        world[2] - camera[14],
    ];
    let dot = |column: usize| {
        p[0] * camera[column * 4] + p[1] * camera[column * 4 + 1] + p[2] * camera[column * 4 + 2]
    };
    let z = -dot(2);
    if !z.is_finite() || z <= 0.0 {
        return None;
    }
    let fy = height as f32 / (2.0 * (fovy * 0.5).tan());
    Some([
        fy * dot(0) / z + width as f32 * 0.5,
        -fy * dot(1) / z + height as f32 * 0.5,
        z,
    ])
}

pub fn visibility(
    scene: &GeometryScene,
    target: usize,
    reference: usize,
    pixel: usize,
) -> Visibility {
    let source_depth = scene.depth[target][pixel];
    if source_depth <= 0.0 {
        return Visibility::Unknown;
    }
    let p = &scene.position[target][pixel * 3..pixel * 3 + 3];
    let Some([u, v, z]) = project(
        [p[0], p[1], p[2]],
        &scene.world_from_view[reference],
        scene.fovy[reference],
        scene.width,
        scene.height,
    ) else {
        return Visibility::OutOfView;
    };
    if u < 0.0 || v < 0.0 || u >= scene.width as f32 || v >= scene.height as f32 {
        return Visibility::OutOfView;
    }
    let depth = scene.depth[reference][v.floor() as usize * scene.width + u.floor() as usize];
    if depth <= 0.0 {
        return Visibility::Unknown;
    }
    // A point behind the reference surface is occluded. A point in front of it may be
    // a disocclusion/boundary sampling mismatch, so it remains unknown.
    let tolerance = 0.02f32.max(0.01 * z);
    if (z - depth).abs() <= tolerance {
        Visibility::Visible
    } else if z > depth + tolerance {
        Visibility::Occluded
    } else {
        Visibility::Unknown
    }
}

#[derive(Debug, Serialize)]
pub struct PairVisibility {
    pub room_seed: u64,
    pub split: Split,
    pub target: usize,
    pub reference: usize,
    pub visible: usize,
    pub occluded: usize,
    pub out_of_view: usize,
    pub unknown: usize,
}

#[derive(Debug, Serialize)]
pub struct GeometryAudit {
    pub dataset_id: String,
    pub rooms: usize,
    pub valid_source_pixels: usize,
    pub max_self_reprojection_pixels: f32,
    pub max_self_depth_error: f32,
    pub visible: usize,
    pub occluded: usize,
    pub out_of_view: usize,
    pub unknown: usize,
    pub tolerance: String,
    /// All directed pairs; the legacy aggregate above measures target -> next view only.
    pub pairs: Vec<PairVisibility>,
}
/// Uses geometry exclusively for verification/evaluation, never as model input or an RGB loss target.
pub fn audit(dir: &Path) -> Result<GeometryAudit> {
    let manifest = open_dataset(dir)?;
    let mut out = GeometryAudit {
        dataset_id: manifest.dataset_id,
        rooms: manifest.scenes.len(),
        valid_source_pixels: 0,
        max_self_reprojection_pixels: 0.0,
        max_self_depth_error: 0.0,
        visible: 0,
        occluded: 0,
        out_of_view: 0,
        unknown: 0,
        tolerance:
            "nearest pixel; max(0.02 m, 1% projected depth); front-of-depth mismatch is unknown"
                .into(),
        pairs: Vec::new(),
    };
    for entry in &manifest.scenes {
        let g = load_geometry(&dir.join("raw").join(&entry.file))?;
        for target in 0..g.depth.len() {
            let mut pairs: Vec<_> = (0..g.depth.len())
                .filter(|&r| r != target)
                .map(|reference| PairVisibility {
                    room_seed: entry.seed,
                    split: entry.split,
                    target,
                    reference,
                    visible: 0,
                    occluded: 0,
                    out_of_view: 0,
                    unknown: 0,
                })
                .collect();
            for p in 0..g.width * g.height {
                if g.depth[target][p] > 0.0 {
                    let pos = &g.position[target][p * 3..p * 3 + 3];
                    let projected = project(
                        [pos[0], pos[1], pos[2]],
                        &g.world_from_view[target],
                        g.fovy[target],
                        g.width,
                        g.height,
                    )
                    .ok_or_else(|| {
                        anyhow::anyhow!("valid depth projects behind its source camera")
                    })?;
                    let error = ((projected[0] - (p % g.width) as f32 - 0.5).powi(2)
                        + (projected[1] - (p / g.width) as f32 - 0.5).powi(2))
                    .sqrt();
                    out.max_self_reprojection_pixels = out.max_self_reprojection_pixels.max(error);
                    out.max_self_depth_error = out
                        .max_self_depth_error
                        .max((projected[2] - g.depth[target][p]).abs());
                    out.valid_source_pixels += 1;
                }
                for pair in &mut pairs {
                    let label = visibility(&g, target, pair.reference, p);
                    match label {
                        Visibility::Visible => pair.visible += 1,
                        Visibility::Occluded => pair.occluded += 1,
                        Visibility::OutOfView => pair.out_of_view += 1,
                        Visibility::Unknown => pair.unknown += 1,
                    }
                    if pair.reference == (target + 1) % g.depth.len() {
                        match label {
                            Visibility::Visible => out.visible += 1,
                            Visibility::Occluded => out.occluded += 1,
                            Visibility::OutOfView => out.out_of_view += 1,
                            Visibility::Unknown => out.unknown += 1,
                        }
                    }
                }
            }
            out.pairs.extend(pairs);
        }
    }
    ensure!(out.valid_source_pixels > 0, "no valid geometry");
    ensure!(
        out.max_self_reprojection_pixels < 0.1 && out.max_self_depth_error < 0.01,
        "camera/depth/position convention failed: reprojection={} px, depth={} m",
        out.max_self_reprojection_pixels,
        out.max_self_depth_error
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn visibility_distinguishes_occlusion_unknown_and_out_of_view() {
        let identity = [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];
        let mut scene = GeometryScene {
            width: 1,
            height: 1,
            depth: vec![vec![2.], vec![2.]],
            position: vec![vec![0., 0., -2.], vec![0., 0., -2.]],
            world_from_view: vec![identity; 2],
            fovy: vec![1.; 2],
        };
        assert_eq!(visibility(&scene, 0, 1, 0), Visibility::Visible);
        scene.depth[1][0] = 1.;
        assert_eq!(visibility(&scene, 0, 1, 0), Visibility::Occluded);
        scene.depth[1][0] = 0.;
        assert_eq!(visibility(&scene, 0, 1, 0), Visibility::Unknown);
        scene.depth[1][0] = 3.;
        assert_eq!(visibility(&scene, 0, 1, 0), Visibility::Unknown);
        scene.world_from_view[1][12] = 10.;
        assert_eq!(visibility(&scene, 0, 1, 0), Visibility::OutOfView);
        // Camera rotated +90 degrees about world Y: its forward vector is world -X.
        let rotated = [
            0., 0., -1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
        ];
        assert_eq!(
            project([-2., 0., 0.], &rotated, 1., 2, 2),
            Some([1., 1., 2.])
        );
    }
}

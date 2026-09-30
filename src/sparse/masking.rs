//! Stateless target masks. The compact policy retains a connected spatial island.
use anyhow::{Result, ensure};
use burn_vjepa::SparseTokenMask;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskPattern {
    #[default]
    Random,
    /// Keep a compact, four-connected set; hide its complement. This is not
    /// V-JEPA's multi-block sampler and does not imply protocol equivalence.
    VisibleBlock,
}

pub fn mask(
    grid: [usize; 2],
    ratio: f32,
    seed: u64,
    step: usize,
    pattern: MaskPattern,
) -> Result<SparseTokenMask> {
    let [h, w] = grid;
    ensure!(h > 0 && w > 0, "empty mask grid");
    let random = crate::encoder::visible_mask(h * w, ratio, seed, step)?;
    if pattern == MaskPattern::Random {
        return Ok(random);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(
        seed.wrapping_add((step as u64).wrapping_mul(0x9e3779b97f4a7c15)),
    );
    let cy = rng.random_range(0..h);
    let cx = rng.random_range(0..w);
    let mut ids: Vec<_> = (0..h * w).collect();
    // Manhattan balls clipped by the image, with deterministic random ties.
    // Every retained point has an earlier four-neighbour toward the center.
    let tie: Vec<u64> = ids.iter().map(|_| rng.random()).collect();
    ids.sort_by_key(|&i| ((i / w).abs_diff(cy) + (i % w).abs_diff(cx), tie[i]));
    ids.truncate(random.len());
    SparseTokenMask::new(ids, h * w)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_masks_replay_have_equal_counts_and_are_connected() {
        for grid in [[16, 16], [3, 7], [1, 8]] {
            for step in 0..25 {
                let a = mask(grid, 0.9, 721, step, MaskPattern::VisibleBlock).unwrap();
                let b = mask(grid, 0.9, 721, step, MaskPattern::VisibleBlock).unwrap();
                let r = mask(grid, 0.9, 721, step, MaskPattern::Random).unwrap();
                assert_eq!(a.indices(), b.indices());
                assert_eq!(a.len(), r.len());
                let mut reached = vec![a.indices()[0]];
                loop {
                    let old = reached.len();
                    for &i in a.indices() {
                        if !reached.contains(&i)
                            && reached.iter().any(|&j| {
                                (i / grid[1]).abs_diff(j / grid[1])
                                    + (i % grid[1]).abs_diff(j % grid[1])
                                    == 1
                            })
                        {
                            reached.push(i);
                        }
                    }
                    if old == reached.len() {
                        break;
                    }
                }
                assert_eq!(reached.len(), a.len());
            }
        }
        assert!(mask([0, 16], 0.9, 1, 0, MaskPattern::Random).is_err());
        assert!(mask([16, 16], 1., 1, 0, MaskPattern::VisibleBlock).is_err());
        assert_ne!(
            mask([16, 16], 0.9, 721, 0, MaskPattern::VisibleBlock)
                .unwrap()
                .indices(),
            mask([16, 16], 0.9, 721, 1, MaskPattern::VisibleBlock)
                .unwrap()
                .indices()
        );
    }
}

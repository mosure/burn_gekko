//! Deterministic cluster bootstrap. Cluster means, not individual correlated pixels, are resampled.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interval {
    pub mean: f64,
    pub low: f64,
    pub high: f64,
    pub clusters: usize,
    pub replicates: usize,
}
pub fn bootstrap_mean(cluster_means: &[f64], seed: u64) -> Result<Interval> {
    ensure!(
        !cluster_means.is_empty() && cluster_means.iter().all(|x| x.is_finite()),
        "invalid bootstrap observations"
    );
    // SplitMix64 with rejection sampling: explicit portable RNG, no library-version dependence.
    let mut state = seed;
    let mut index = || {
        let n = cluster_means.len() as u64;
        loop {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = state;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            if x >= n.wrapping_neg() % n {
                break (x % n) as usize;
            }
        }
    };
    let n = cluster_means.len();
    let count = 10000;
    let mut means = (0..count)
        .map(|_| (0..n).map(|_| cluster_means[index()]).sum::<f64>() / n as f64)
        .collect::<Vec<_>>();
    means.sort_by(f64::total_cmp);
    let quantile = |q: f64| {
        let x = q * (count - 1) as f64;
        let lo = x.floor() as usize;
        let hi = x.ceil() as usize;
        means[lo] + (means[hi] - means[lo]) * x.fract()
    };
    Ok(Interval {
        mean: cluster_means.iter().sum::<f64>() / n as f64,
        low: quantile(0.025),
        high: quantile(0.975),
        clusters: n,
        replicates: count,
    })
}

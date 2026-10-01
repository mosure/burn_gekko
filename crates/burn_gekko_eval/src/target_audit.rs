//! Native coverage audit of immutable renderer-supervised target caches.
use anyhow::{Result, ensure};
use burn_gekko_data::{
    sha256_file,
    view_targets::{Cache, labels},
    write_json,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub dataset: PathBuf,
    pub cache: PathBuf,
    pub output: PathBuf,
}
#[derive(Default, Serialize)]
struct Coverage {
    rooms: usize,
    directed_pairs: usize,
    empty_pairs: usize,
    valid_queries: usize,
    minimum_valid_queries: usize,
    maximum_valid_queries: usize,
    mean_valid_queries: f64,
}
pub fn audit(c: &Config) -> Result<serde_json::Value> {
    ensure!(!c.output.exists(), "preserve existing target audit");
    let cache = Cache::load(&c.cache, &c.dataset)?;
    let mut by_split: BTreeMap<String, Coverage> = BTreeMap::new();
    for (i, room) in cache.manifest.rooms.iter().enumerate() {
        let split = by_split
            .entry(format!("{:?}", room.split))
            .or_insert_with(|| Coverage {
                minimum_valid_queries: usize::MAX,
                ..Default::default()
            });
        split.rooms += 1;
        for a in 0..cache.manifest.views {
            for b in 0..cache.manifest.views {
                if a == b {
                    continue;
                }
                let (_, n) = labels(
                    &cache.pair(i, a, b)?,
                    cache.manifest.grid,
                    cache.manifest.patch,
                )?;
                split.directed_pairs += 1;
                split.empty_pairs += usize::from(n == 0);
                split.valid_queries += n;
                split.minimum_valid_queries = split.minimum_valid_queries.min(n);
                split.maximum_valid_queries = split.maximum_valid_queries.max(n);
            }
        }
    }
    for split in by_split.values_mut() {
        ensure!(split.directed_pairs > 0, "no directed pairs");
        split.mean_valid_queries = split.valid_queries as f64 / split.directed_pairs as f64;
    }
    ensure!(
        by_split.values().map(|s| s.valid_queries).sum::<usize>() == cache.manifest.valid_queries,
        "valid-query count differs"
    );
    let report = serde_json::json!({"schema":1,"dataset_id":cache.manifest.dataset_id,"cache_manifest_sha256":sha256_file(&c.cache.join("manifest.json"))?,"target_sha256":cache.manifest.targets_sha256,"state_counts":cache.manifest.state_counts,"valid_queries":cache.manifest.valid_queries,"by_split":by_split,"policy":cache.manifest.policy});
    write_json(&c.output, &report)?;
    Ok(report)
}

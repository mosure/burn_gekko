//! CPU integrity/coverage audit of a prepared target cache; no model inference.
use anyhow::{Context, Result, ensure};
use burn_gekko_data::view_targets::{Cache, State, labels};
use std::{collections::BTreeMap, path::Path};
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(
        args.len() == 3,
        "usage: view_targets_audit CACHE DATASET OUTPUT.json"
    );
    let cache = Cache::load(Path::new(&args[0]), Path::new(&args[1]))?;
    let mut states = [0; 4];
    let mut valid = 0;
    let mut by_split: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut rows = Vec::new();
    for (i, room) in cache.manifest.rooms.iter().enumerate() {
        for a in 0..cache.manifest.views {
            for b in 0..cache.manifest.views {
                if a == b {
                    continue;
                }
                let points = cache.pair(i, a, b)?;
                for p in &points {
                    states[match p.state {
                        State::Unknown => 0,
                        State::Visible => 1,
                        State::Occluded => 2,
                        State::OutOfView => 3,
                    }] += 1;
                }
                let count = labels(&points, cache.manifest.grid, cache.manifest.patch)?.1;
                valid += count;
                by_split
                    .entry(format!("{:?}", room.split))
                    .or_default()
                    .push(count);
                rows.push(serde_json::json!({"seed":room.seed,"split":room.split,"target":a,"reference":b,"valid_queries":count}));
            }
        }
    }
    ensure!(
        states == cache.manifest.state_counts && valid == cache.manifest.valid_queries,
        "cache aggregate counts differ"
    );
    let groups:BTreeMap<_,_>=by_split.into_iter().map(|(split,counts)| {
        let value=serde_json::json!({"directed_pairs":counts.len(),"minimum_valid_queries":counts.iter().min().unwrap(),"empty_pairs":counts.iter().filter(|n|**n==0).count(),"mean_valid_queries":counts.iter().sum::<usize>() as f64/counts.len() as f64});
        (split,value)
    }).collect();
    let output = Path::new(&args[2]);
    ensure!(!output.exists(), "audit output exists");
    ensure!(
        std::fs::canonicalize(output.parent().context("output parent")?)?
            .starts_with(std::fs::canonicalize(".data")?),
        "audit outside .data"
    );
    burn_gekko_data::write_json(
        output,
        &serde_json::json!({"dataset_id":cache.manifest.dataset_id,"target_sha256":cache.manifest.targets_sha256,"cache_manifest_sha256":burn_gekko_data::sha256_file(&Path::new(&args[0]).join("manifest.json"))?,"state_counts":states,"valid_queries":valid,"by_split":groups,"pairs":rows}),
    )?;
    println!("{}", serde_json::to_string_pretty(&groups)?);
    Ok(())
}

//! Immutable compact CPU target cache with data, source and content identities.
use super::*;
use crate::{
    Split, fingerprint, load_geometry, read_dataset_manifest, sha256_file, write_config, write_json,
};
use anyhow::Context;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};
const RECORD_BYTES: usize = 9;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrepareConfig {
    pub dataset: PathBuf,
    pub output: PathBuf,
    pub rooms_per_split: [usize; 3],
    pub patch: usize,
}

#[derive(Serialize, Deserialize)]
pub struct Room {
    pub seed: u64,
    pub split: Split,
    pub shard_sha256: String,
}

#[derive(Serialize, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    pub dataset_id: String,
    pub policy: String,
    pub source_sha256: String,
    pub patch: usize,
    pub grid: [usize; 2],
    pub views: usize,
    pub rooms: Vec<Room>,
    pub targets_sha256: String,
    pub state_counts: [usize; 4],
    pub valid_queries: usize,
}

fn source_identity() -> Result<String> {
    fingerprint(&(
        include_str!("../view_targets.rs"),
        include_str!("cache.rs"),
        include_str!("../geometry.rs"),
        include_str!("../chunk.rs"),
    ))
}

pub fn prepare(c: &PrepareConfig) -> Result<()> {
    ensure!(
        !c.output.exists() && c.rooms_per_split.iter().sum::<usize>() > 0,
        "cache exists or empty selection"
    );
    ensure!(
        fs::canonicalize(
            c.output
                .ancestors()
                .find(|p| p.exists())
                .context("missing output parent")?
        )?
        .starts_with(fs::canonicalize(".data")?),
        "cache must be under .data"
    );
    let m = read_dataset_manifest(&c.dataset)?;
    ensure!(
        c.patch >= 2
            && c.patch.is_multiple_of(2)
            && m.config.width.is_multiple_of(c.patch)
            && m.config.height.is_multiple_of(c.patch),
        "invalid cache patch size"
    );
    fs::create_dir_all(&c.output)?;
    write_config(&c.output.join("config.toml"), c)?;
    let mut file = fs::File::create(c.output.join("targets.bin"))?;
    let grid = [m.config.height / c.patch, m.config.width / c.patch];
    let mut manifest = Manifest {
        schema: 1,
        dataset_id: m.dataset_id.clone(),
        policy: POLICY.into(),
        source_sha256: source_identity()?,
        patch: c.patch,
        grid,
        views: m.config.cameras,
        rooms: Vec::new(),
        targets_sha256: String::new(),
        state_counts: [0; 4],
        valid_queries: 0,
    };
    for (split, count) in [Split::Train, Split::Validation, Split::Test]
        .into_iter()
        .zip(c.rooms_per_split)
    {
        let selected = m
            .scenes
            .iter()
            .filter(|s| s.split == split)
            .take(count)
            .collect::<Vec<_>>();
        ensure!(
            selected.len() == count,
            "insufficient rooms for target cache"
        );
        for entry in selected {
            let path = c.dataset.join("raw").join(&entry.file);
            ensure!(
                sha256_file(&path)? == entry.sha256,
                "geometry shard checksum differs"
            );
            let scene = load_geometry(&path)?;
            ensure!(
                (scene.width, scene.height, scene.depth.len())
                    == (m.config.width, m.config.height, m.config.cameras),
                "geometry shape differs"
            );
            for a in 0..manifest.views {
                for b in 0..manifest.views {
                    if a == b {
                        continue;
                    }
                    let points = pair(&scene, a, b, c.patch)?;
                    manifest.valid_queries += labels(&points, grid, c.patch)?.1;
                    let mut bytes = Vec::with_capacity(points.len() * RECORD_BYTES);
                    for p in points {
                        let state = match p.state {
                            State::Unknown => 0,
                            State::Visible => 1,
                            State::Occluded => 2,
                            State::OutOfView => 3,
                        };
                        manifest.state_counts[state] += 1;
                        bytes.push(state as u8);
                        for x in p.xy {
                            bytes.extend(x.to_le_bytes());
                        }
                    }
                    file.write_all(&bytes)?;
                }
            }
            manifest.rooms.push(Room {
                seed: entry.seed,
                split,
                shard_sha256: entry.sha256.clone(),
            });
        }
    }
    file.sync_all()?;
    manifest.targets_sha256 = sha256_file(&c.output.join("targets.bin"))?;
    write_json(&c.output.join("manifest.json"), &manifest)?;
    Ok(())
}

pub struct Cache {
    pub manifest: Manifest,
    bytes: Vec<u8>,
}
impl Cache {
    pub fn load(dir: &Path, dataset: &Path) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(&fs::read(dir.join("manifest.json"))?)?;
        let dataset = read_dataset_manifest(dataset)?;
        ensure!(
            manifest.schema == 1
                && manifest.dataset_id == dataset.dataset_id
                && manifest.policy == POLICY
                && manifest.source_sha256 == source_identity()?,
            "target cache identity differs"
        );
        ensure!(
            manifest.views == dataset.config.cameras
                && manifest.patch > 0
                && manifest.grid
                    == [
                        dataset.config.height / manifest.patch,
                        dataset.config.width / manifest.patch
                    ],
            "target cache shape differs"
        );
        let by_seed = dataset
            .scenes
            .iter()
            .map(|s| (s.seed, s))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut seen = std::collections::BTreeSet::new();
        for room in &manifest.rooms {
            let entry = by_seed
                .get(&room.seed)
                .context("cached room absent from dataset")?;
            ensure!(
                seen.insert(room.seed)
                    && entry.split == room.split
                    && entry.sha256 == room.shard_sha256,
                "cached room identity differs"
            );
        }
        let path = dir.join("targets.bin");
        ensure!(
            sha256_file(&path)? == manifest.targets_sha256,
            "target cache checksum differs"
        );
        let bytes = fs::read(path)?;
        ensure!(
            bytes.len()
                == manifest.rooms.len()
                    * manifest.views
                    * (manifest.views - 1)
                    * manifest.grid[0]
                    * manifest.grid[1]
                    * RECORD_BYTES,
            "target cache length differs"
        );
        Ok(Self { manifest, bytes })
    }
    pub fn room_index(&self, seed: u64, split: Split) -> Result<usize> {
        self.manifest
            .rooms
            .iter()
            .position(|r| r.seed == seed && r.split == split)
            .context("room/split absent from target cache")
    }
    pub fn pair(&self, room: usize, target: usize, reference: usize) -> Result<Vec<Point>> {
        let m = &self.manifest;
        ensure!(
            room < m.rooms.len() && target < m.views && reference < m.views && target != reference,
            "invalid cached pair"
        );
        let n = m.grid[0] * m.grid[1];
        let index = (room * m.views * (m.views - 1)
            + target * (m.views - 1)
            + if reference < target {
                reference
            } else {
                reference - 1
            })
            * n
            * RECORD_BYTES;
        self.bytes[index..index + n * RECORD_BYTES]
            .as_chunks::<RECORD_BYTES>()
            .0
            .iter()
            .map(|row| {
                let state = match row[0] {
                    0 => State::Unknown,
                    1 => State::Visible,
                    2 => State::Occluded,
                    3 => State::OutOfView,
                    _ => anyhow::bail!("invalid cached state"),
                };
                let xy = [
                    f32::from_le_bytes(row[1..5].try_into()?),
                    f32::from_le_bytes(row[5..9].try_into()?),
                ];
                ensure!(xy.iter().all(|x| x.is_finite()), "nonfinite cached point");
                Ok(Point { state, xy })
            })
            .collect()
    }
}

use crate::{
    CaptureConfig, GENERATOR, LEGACY_GENERATOR, PILOT02_GENERATOR, PILOT05_GENERATOR, SCHEMA,
    Split, load_geometry, load_rgb, read_config_snapshot, write_config,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::Read,
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub fn fingerprint<T: Serialize>(value: &T) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneEntry {
    pub file: String,
    pub sha256: String,
    pub seed: u64,
    pub split: Split,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetManifest {
    pub schema: u32,
    pub dataset_id: String,
    pub generator: String,
    pub binary_sha256: String,
    pub config: CaptureConfig,
    pub scenes: Vec<SceneEntry>,
}

fn safe_file(name: &str) -> bool {
    let mut it = Path::new(name).components();
    matches!(it.next(), Some(Component::Normal(_))) && it.next().is_none()
}

/// Validate the manifest identity and split layout without decoding every shard.
/// Consumers must call `load_dataset_rgb` for every selected scene before use.
pub fn read_dataset_manifest(dir: &Path) -> Result<DatasetManifest> {
    let m: DatasetManifest = serde_json::from_slice(&fs::read(dir.join("manifest.json"))?)?;
    m.config.validate()?;
    ensure!(
        m.schema == SCHEMA
            && [
                GENERATOR,
                PILOT05_GENERATOR,
                PILOT02_GENERATOR,
                LEGACY_GENERATOR
            ]
            .contains(&m.generator.as_str()),
        "unsupported dataset provenance"
    );
    ensure!(
        m.dataset_id == fingerprint(&(m.schema, &m.generator, &m.binary_sha256, &m.config))?,
        "dataset identity mismatch"
    );
    ensure!(m.scenes.len() == m.config.scenes(), "incomplete dataset");
    let mut seeds = BTreeSet::new();
    let mut files = BTreeSet::new();
    for (i, scene) in m.scenes.iter().enumerate() {
        ensure!(
            safe_file(&scene.file) && files.insert(&scene.file),
            "invalid or duplicate shard path"
        );
        ensure!(scene.split == m.config.split(i)?, "split manifest mismatch");
        ensure!(
            seeds.insert(scene.seed),
            "room seed appears in multiple records/splits"
        );
        ensure!(
            scene.seed == m.config.seed + i as u64,
            "room seed sequence differs from capture configuration"
        );
    }
    Ok(m)
}

/// Verify membership, file checksum and decoded metadata for one selected shard.
pub fn load_dataset_rgb(
    dir: &Path,
    manifest: &DatasetManifest,
    scene: &SceneEntry,
) -> Result<crate::RgbScene> {
    let index = scene
        .seed
        .checked_sub(manifest.config.seed)
        .and_then(|i| usize::try_from(i).ok())
        .context("scene outside manifest")?;
    let expected = manifest
        .scenes
        .get(index)
        .context("scene outside manifest")?;
    ensure!(
        safe_file(&scene.file)
            && scene.file == expected.file
            && scene.sha256 == expected.sha256
            && scene.split == expected.split,
        "scene differs from manifest"
    );
    let path = dir.join("raw").join(&scene.file);
    ensure!(
        fs::symlink_metadata(&path)?.file_type().is_file(),
        "shard must be a regular immutable file"
    );
    ensure!(
        sha256_file(&path)? == scene.sha256,
        "shard checksum mismatch: {}",
        scene.file
    );
    let rgb = load_rgb(&path)?;
    ensure!(
        rgb.seed == scene.seed
            && rgb.width == manifest.config.width
            && rgb.height == manifest.config.height
            && rgb.views.len() == manifest.config.cameras,
        "shard metadata mismatch"
    );
    Ok(rgb)
}

/// Full cache audit, used by capture/reuse verification and explicit inspection.
pub fn open_dataset(dir: &Path) -> Result<DatasetManifest> {
    let manifest = read_dataset_manifest(dir)?;
    for scene in &manifest.scenes {
        load_dataset_rgb(dir, &manifest, scene)?;
    }
    Ok(manifest)
}

struct Lock(PathBuf);
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// A successful capture is validated then atomically made visible. Failed staging data stays diagnostic.
pub fn generate(data_root: &Path, binary: &Path, config: &CaptureConfig) -> Result<PathBuf> {
    config.validate()?;
    let binary = fs::canonicalize(binary).context("build tools/zeroverse_capture first")?;
    let identity = Command::new(&binary).arg("--identity").output()?;
    ensure!(
        identity.status.success() && String::from_utf8_lossy(&identity.stdout).trim() == GENERATOR,
        "capture binary identity mismatch; rebuild the latest pinned capture tool"
    );
    let hash = sha256_file(&binary)?;
    let id = fingerprint(&(SCHEMA, GENERATOR, &hash, config))?;
    let base = data_root.join("datasets");
    fs::create_dir_all(&base)?;
    let destination = base.join(&id);
    if destination.exists() {
        open_dataset(&destination)?;
        return Ok(destination);
    }
    let lock_path = base.join(format!("{id}.lock"));
    let _handle = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .context("capture already running, or a stale lock needs inspection")?;
    let _lock = Lock(lock_path);
    let stage = base.join(format!(".{id}.partial-{}", std::process::id()));
    fs::create_dir(&stage).context("existing partial capture; inspect it before retrying")?;
    fs::create_dir(stage.join("raw"))?;
    write_config(&stage.join("capture.toml"), config)?;
    let stage_abs = fs::canonicalize(&stage)?;
    let log = File::create(stage.join("capture.log"))?;
    let mut child = Command::new(&binary)
        .arg("--config")
        .arg(stage_abs.join("capture.toml"))
        .arg("--output")
        .arg(stage_abs.join("raw"))
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            ensure!(
                status.success(),
                "capture failed ({status}); inspect {}",
                stage.join("capture.log").display()
            );
            break;
        }
        if start.elapsed() > Duration::from_secs(config.timeout_secs) {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!(
                "capture timeout; incomplete output preserved at {}",
                stage.display()
            );
        }
        thread::sleep(Duration::from_millis(100));
    }
    finalize_capture(&stage, &destination, config, hash, id)
}

/// Explicitly validate an already captured staging directory after fixing an adapter issue.
/// The original binary and config fingerprint must match; generation is never silently retried.
pub fn recover_capture(data_root: &Path, binary: &Path, stage: &Path) -> Result<PathBuf> {
    let config: CaptureConfig = read_config_snapshot(stage, "capture")?;
    config.validate()?;
    let output = Command::new(binary).arg("--identity").output()?;
    ensure!(
        output.status.success() && String::from_utf8_lossy(&output.stdout).trim() == GENERATOR,
        "generator identity mismatch"
    );
    let hash = sha256_file(binary)?;
    let id = fingerprint(&(SCHEMA, GENERATOR, &hash, &config))?;
    ensure!(
        stage
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.starts_with(&format!(".{id}.partial-"))),
        "staging config/binary fingerprint mismatch"
    );
    let base = data_root.join("datasets");
    ensure!(
        fs::canonicalize(stage.parent().context("stage parent")?)? == fs::canonicalize(&base)?,
        "staging must belong to dataset cache"
    );
    let lock_path = base.join(format!("{id}.lock"));
    let _handle = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .context("capture still locked")?;
    let _lock = Lock(lock_path);
    let destination = base.join(&id);
    ensure!(!destination.exists(), "completed cache already exists");
    write_json(
        &stage.join("recovery.json"),
        &serde_json::json!({"explicit_revalidation":true,"generator":GENERATOR,"binary_sha256":hash}),
    )?;
    finalize_capture(stage, &destination, &config, hash, id)
}

fn finalize_capture(
    stage: &Path,
    destination: &Path,
    config: &CaptureConfig,
    hash: String,
    id: String,
) -> Result<PathBuf> {
    let mut files: Vec<_> = fs::read_dir(stage.join("raw"))?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<_>>()?;
    files.sort();
    ensure!(
        files.len() == config.scenes(),
        "expected exactly one shard per room"
    );
    let mut scenes = Vec::new();
    for (i, path) in files.iter().enumerate() {
        let rgb = load_rgb(path)?;
        load_geometry(path).context("capture geometry contract")?;
        scenes.push(SceneEntry {
            file: path
                .file_name()
                .unwrap()
                .to_str()
                .context("non-UTF8 shard name")?
                .into(),
            sha256: sha256_file(path)?,
            seed: rgb.seed,
            split: config.split(i)?,
        });
    }
    let manifest = DatasetManifest {
        schema: SCHEMA,
        dataset_id: id,
        generator: GENERATOR.into(),
        binary_sha256: hash,
        config: config.clone(),
        scenes,
    };
    write_json(&stage.join("manifest.json"), &manifest)?;
    open_dataset(stage)?;
    crate::audit(stage)?;
    fs::rename(stage, destination)?;
    Ok(destination.to_path_buf())
}

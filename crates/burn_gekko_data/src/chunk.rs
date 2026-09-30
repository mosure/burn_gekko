use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};
use std::{fs::File, io::Read, path::Path};

const MAX_CHUNK_BYTES: u64 = 128 * 1024 * 1024;

/// RGB-only training API. Geometry is intentionally exposed through a separate loader.
#[derive(Clone, Debug)]
pub struct RgbScene {
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    /// One HWC, sRGB float32 image per camera, all at the same instant.
    pub views: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct GeometryScene {
    pub width: usize,
    pub height: usize,
    pub depth: Vec<Vec<f32>>,
    pub position: Vec<Vec<f32>>,
    /// Bevy column-major world_from_view, looking down local -Z.
    pub world_from_view: Vec<[f32; 16]>,
    pub fovy: Vec<f32>,
}

fn bytes(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader: Box<dyn Read> = if path.extension().is_some_and(|e| e == "zst") {
        Box::new(zstd::stream::read::Decoder::new(file)?)
    } else {
        Box::new(file)
    };
    let mut out = Vec::new();
    reader.take(MAX_CHUNK_BYTES + 1).read_to_end(&mut out)?;
    ensure!(
        out.len() as u64 <= MAX_CHUNK_BYTES,
        "chunk exceeds preflight byte limit"
    );
    Ok(out)
}

fn floats(tensors: &SafeTensors<'_>, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let t = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    ensure!(
        t.dtype() == Dtype::F32 && t.shape() == expected,
        "{name}: expected F32 {expected:?}, got {:?} {:?}",
        t.dtype(),
        t.shape()
    );
    let values: Vec<_> = t
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite {name}");
    Ok(values)
}

fn shape(t: &SafeTensors<'_>) -> Result<(usize, usize, usize)> {
    let rgb = t
        .tensor("color")
        .context("raw color required; JPEG and legacy captures are not supported")?;
    let s = rgb.shape();
    ensure!(
        s.len() == 6 && s[0] == 1 && s[1] == 1 && s[5] == 3,
        "expected color [1,1,cameras,height,width,3], got {s:?}"
    );
    ensure!(
        (2..=4).contains(&s[2]) && (32..=512).contains(&s[3]) && (32..=512).contains(&s[4]),
        "chunk dimensions exceed workstation capture contract"
    );
    let encoding = t.tensor("color_encoding")?;
    ensure!(
        encoding.dtype() == Dtype::U8 && encoding.shape() == [1] && encoding.data() == [2],
        "explicit sRGB color_encoding=2 required"
    );
    let precision = t.tensor("annotation_precision")?;
    ensure!(
        precision.dtype() == Dtype::U8 && precision.shape() == [1] && precision.data() == [1],
        "float32 annotations required"
    );
    Ok((s[2], s[3], s[4]))
}

pub fn load_rgb(path: &Path) -> Result<RgbScene> {
    let buf = bytes(path)?;
    let t = SafeTensors::deserialize(&buf)?;
    let (views, height, width) = shape(&t)?;
    let rgb = floats(&t, "color", &[1, 1, views, height, width, 3])?;
    ensure!(
        rgb.iter().all(|&x| (-0.001..=1.001).contains(&x)),
        "RGB outside sRGB [0,1]"
    );
    let manifest = t.tensor("indoor_manifest_0")?;
    ensure!(
        manifest.dtype() == Dtype::U8,
        "manifest must be UTF-8 bytes"
    );
    let manifest: serde_json::Value = serde_json::from_slice(manifest.data())?;
    let seed = manifest["seed"]
        .as_u64()
        .context("indoor manifest lacks seed")?;
    Ok(RgbScene {
        seed,
        width,
        height,
        views: rgb
            .chunks_exact(height * width * 3)
            .map(<[f32]>::to_vec)
            .collect(),
    })
}

pub fn load_geometry(path: &Path) -> Result<GeometryScene> {
    let buf = bytes(path)?;
    let t = SafeTensors::deserialize(&buf)?;
    let (views, height, width) = shape(&t)?;
    let depth = floats(&t, "depth", &[1, 1, views, height, width, 1])?;
    let mut position = floats(&t, "position", &[1, 1, views, height, width, 3])?;
    // Published Zeroverse exports AABB-normalized positions, including its float32 geometry path.
    let aabb = floats(&t, "aabb", &[1, 2, 3])?;
    ensure!(
        (0..3).all(|axis| aabb[axis + 3] > aabb[axis]),
        "degenerate scene AABB"
    );
    let unclamped = if let Ok(metadata) = t.tensor("indoor_render_metadata_0") {
        ensure!(
            metadata.dtype() == Dtype::U8,
            "render metadata must be UTF-8 bytes"
        );
        let value: serde_json::Value = serde_json::from_slice(metadata.data())?;
        value["capture_engine"]
            .as_str()
            .is_some_and(|s| s.split(';').any(|part| part == "position=2"))
    } else {
        false
    };
    // 0.22 preserves visible surfaces outside the reconstruction AABB. Its position=2
    // contract is affine, not clamped to [0,1]; use the same inverse transform.
    ensure!(
        unclamped || position.iter().all(|&v| (-1e-5..=1.00001).contains(&v)),
        "position outside legacy normalized AABB without position=2 metadata"
    );
    for (i, value) in position.iter_mut().enumerate() {
        let axis = i % 3;
        *value = aabb[axis] + *value * (aabb[axis + 3] - aabb[axis]);
    }
    let matrices = floats(&t, "world_from_view", &[1, 1, views, 4, 4])?;
    let fovy = floats(&t, "fovy", &[1, 1, views, 1])?;
    ensure!(
        fovy.iter().all(|&f| f > 0.0 && f < std::f32::consts::PI),
        "invalid fovy"
    );
    Ok(GeometryScene {
        width,
        height,
        depth: depth
            .chunks_exact(height * width)
            .map(<[f32]>::to_vec)
            .collect(),
        position: position
            .chunks_exact(height * width * 3)
            .map(<[f32]>::to_vec)
            .collect(),
        world_from_view: matrices.as_chunks::<16>().0.to_vec(),
        fovy,
    })
}

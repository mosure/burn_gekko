//! Minimal, strict little-endian float NPY reader for the frozen benchmark NPZ files.
use anyhow::{Context, Result, bail, ensure};
use std::{fs::File, io::Read, path::Path};
pub struct Npz {
    archive: zip::ZipArchive<File>,
}
impl Npz {
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            archive: zip::ZipArchive::new(File::open(path)?)?,
        })
    }
    pub fn read(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f64>)> {
        let mut file = self.archive.by_name(&format!("{name}.npy"))?;
        ensure!(
            file.size() <= 128 * 1024 * 1024,
            "oversized benchmark array"
        );
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        parse(&bytes)
    }
}
fn parse(bytes: &[u8]) -> Result<(Vec<usize>, Vec<f64>)> {
    ensure!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "invalid NPY magic"
    );
    let (start, length) = match bytes[6] {
        1 => (10, u16::from_le_bytes(bytes[8..10].try_into()?) as usize),
        2 | 3 => {
            ensure!(bytes.len() >= 12, "short NPY header");
            (12, u32::from_le_bytes(bytes[8..12].try_into()?) as usize)
        }
        _ => bail!("unsupported NPY version"),
    };
    let h = std::str::from_utf8(
        bytes
            .get(start..start + length)
            .context("truncated NPY header")?,
    )?;
    ensure!(
        h.contains("'fortran_order': False"),
        "Fortran order unsupported"
    );
    let dims = h
        .split("'shape':")
        .nth(1)
        .context("missing NPY shape")?
        .split('(')
        .nth(1)
        .context("invalid NPY shape")?
        .split(')')
        .next()
        .unwrap();
    let shape = dims
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let width = if h.contains("'<f8'") {
        8
    } else if h.contains("'<f4'") {
        4
    } else {
        bail!("unsupported NPY float encoding: {h}")
    };
    let data = &bytes[start + length..];
    let count = shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .context("array shape overflow")?;
    ensure!(
        count.checked_mul(width) == Some(data.len()),
        "NPY size mismatch"
    );
    let values = data
        .chunks_exact(width)
        .map(|x| {
            if width == 8 {
                f64::from_le_bytes(x.try_into().unwrap())
            } else {
                f32::from_le_bytes(x.try_into().unwrap()) as f64
            }
        })
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite benchmark geometry"
    );
    Ok((shape, values))
}

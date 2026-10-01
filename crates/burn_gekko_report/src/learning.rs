//! One selected phase's learning curves, shared by the page and vector PDF plot.
use crate::artifact::{Source, number, record};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{fs, path::Path};

fn means(rows: &[Value], start: u64, end: u64) -> Result<Vec<(f64, f64, f64)>> {
    let rows = rows
        .iter()
        .filter(|r| r["step"].as_u64().is_some_and(|s| s > start && s <= end))
        .collect::<Vec<_>>();
    ensure!(
        end > start && rows.len() as u64 == end - start,
        "incomplete learning-curve phase"
    );
    for (i, row) in rows.iter().enumerate() {
        ensure!(
            row["step"] == start + i as u64 + 1,
            "unordered learning-curve phase"
        );
    }
    rows.chunks(50)
        .map(|chunk| {
            let mean = |key| -> Result<f64> {
                Ok(chunk
                    .iter()
                    .map(|r| number(r, key))
                    .collect::<Result<Vec<_>>>()?
                    .iter()
                    .sum::<f64>()
                    / chunk.len() as f64)
            };
            Ok((
                chunk.last().unwrap()["step"]
                    .as_u64()
                    .context("training step")? as f64,
                mean("cross")?,
                mean("monocular")?,
            ))
        })
        .collect()
}

pub fn write(
    run: &Path,
    start: u64,
    end: u64,
    out: &Path,
    sources: &mut Vec<Source>,
) -> Result<()> {
    let file = run.join("metrics.jsonl");
    record(&file, sources)?;
    let rows = fs::read_to_string(file)?
        .lines()
        .map(|r| Ok(serde_json::from_str::<Value>(r)?))
        .collect::<Result<Vec<_>>>()?;
    let points = means(&rows, start, end)?;
    let report_path = run.join("report.json");
    record(&report_path, sources)?;
    let report: Value = serde_json::from_slice(&fs::read(report_path)?)?;
    let mut validation = Vec::new();
    if let Some(probes) = report["probes"].as_array() {
        for p in probes {
            let step = p["step"].as_u64().context("validation step")?;
            if (start..=end).contains(&step) {
                validation.push((step as f64, number(p, "cross_mse")?));
            }
        }
        ensure!(
            validation.windows(2).all(|v| v[0].0 < v[1].0),
            "unordered validation probes"
        );
    }
    let mut csv = String::from("step,cross,monocular\n");
    for (x, c, m) in &points {
        csv.push_str(&format!("{x},{c:.12},{m:.12}\n"));
    }
    fs::write(out.join("media/training.csv"), csv)?;
    let mut csv = String::from("step,cross\n");
    for (x, c) in &validation {
        csv.push_str(&format!("{x},{c:.12}\n"));
    }
    fs::write(out.join("media/validation.csv"), csv)?;
    burn_gekko_data::write_json(&out.join("media/training.json"), &points)?;
    burn_gekko_data::write_json(&out.join("media/validation.json"), &validation)?;

    let min = points
        .iter()
        .flat_map(|p| [p.1, p.2])
        .chain(validation.iter().map(|p| p.1))
        .fold(f64::INFINITY, f64::min);
    let max = points
        .iter()
        .flat_map(|p| [p.1, p.2])
        .chain(validation.iter().map(|p| p.1))
        .fold(f64::NEG_INFINITY, f64::max);
    let lo = (min * 0.95).max(0.);
    let hi = (max * 1.05).max(lo + 1e-6);
    let coord = |x: f64, y: f64| {
        (
            75. + (x - start as f64) / (end - start) as f64 * 765.,
            280. - (y - lo) / (hi - lo) * 215.,
        )
    };
    let mut paths = Vec::new();
    for (series, color, dashed) in [
        (
            points.iter().map(|p| (p.0, p.1)).collect::<Vec<_>>(),
            "#148c78",
            false,
        ),
        (
            points.iter().map(|p| (p.0, p.2)).collect(),
            "#db7957",
            false,
        ),
        (validation.clone(), "#426bc3", true),
    ] {
        let mut path = String::new();
        let mut marks = String::new();
        for (i, (x, y)) in series.iter().enumerate() {
            let (x, y) = coord(*x, *y);
            path.push_str(&format!(
                "{} {x:.2} {y:.2} ",
                if i == 0 { "M" } else { "L" }
            ));
            if dashed || series.len() == 1 {
                marks.push_str(&format!(
                    "<circle cx='{x:.2}' cy='{y:.2}' r='3' fill='{color}'/>"
                ));
            }
        }
        paths.push(format!(
            "<path d='{path}' stroke='{color}' stroke-width='2.5' fill='none' {}/>{marks}",
            if dashed { "stroke-dasharray='6 4'" } else { "" }
        ));
    }
    let mid = (hi + lo) / 2.;
    let xm = (start + end) as f64 / 2.;
    let legend = if validation.is_empty() {
        ""
    } else {
        "<text x='690' y='28' fill='#426bc3'>Validation cross</text>"
    };
    let svg = format!(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 900 350' role='img' aria-label='Training and validation latent mean squared error'><rect width='900' height='350' fill='#fff'/><g font-family='sans-serif' font-size='14' fill='#25354b'><text x='75' y='28'>Latent MSE · lower is better</text><text x='400' y='28' fill='#148c78'>Cross-view train</text><text x='545' y='28' fill='#db7957'>Monocular train</text>{legend}<path d='M75 55V280H840' fill='none' stroke='#8997ab'/><text x='16' y='68'>{hi:.3}</text><text x='16' y='178'>{mid:.3}</text><text x='16' y='284'>{lo:.3}</text><text x='75' y='306'>{start}</text><text x='430' y='306'>{xm:.0}</text><text x='795' y='306'>{end}</text><text x='350' y='335'>Optimizer update in selected phase</text></g>{}</svg>",
        paths.join("")
    );
    fs::write(out.join("media/training.svg"), svg)?;

    // PGFPlots is the standard vector plotting backend for the exported paper;
    // both renderers consume these same native, checkpoint-limited points.
    // Limit labels independently of PGFPlots' automatic density. Long phase
    // indices otherwise overlap at the paper's fixed plot width.
    let span = end - start;
    let mut ticks = vec![
        start,
        start + span / 4,
        start + span / 2,
        end - span / 4,
        end,
    ];
    ticks.dedup();
    let ticks = ticks
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mut tex = format!("\\pgfplotsset{{xmin={start},xmax={end},xtick={{{ticks}}}}}\n");
    tex.push_str(
        r"\begin{tikzpicture}
\begin{axis}[width=\linewidth,height=65mm,xlabel={Optimizer update},ylabel={Latent MSE},grid=major,scaled x ticks=false,legend style={at={(0.5,1.03)},anchor=south,legend columns=3,font=\scriptsize},tick label style={font=\small}]
\addplot[color=teal,mark=none,thick] table[x=step,y=cross,col sep=comma]{media/training.csv};
\addlegendentry{Cross-view train}
\addplot[color=orange,mark=none,thick] table[x=step,y=monocular,col sep=comma]{media/training.csv};
\addlegendentry{Monocular train}
",
    );
    if !validation.is_empty() {
        tex.push_str(r"\addplot[color=blue,dashed,mark=*,mark size=1.5pt] table[x=step,y=cross,col sep=comma]{media/validation.csv};
\addlegendentry{Validation cross}
");
    }
    tex.push_str("\\end{axis}\n\\end{tikzpicture}\n");
    fs::write(out.join("media/training-plot.tex"), tex)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn plotted_losses_exclude_parent_and_postselected_updates() {
        let rows = (1..=4)
            .map(|i| json!({"step":i,"cross":i as f64,"monocular":i as f64+1.}))
            .collect::<Vec<_>>();
        assert_eq!(means(&rows, 1, 3).unwrap(), vec![(3., 2.5, 3.5)]);
        assert!(means(&rows, 0, 5).is_err());
        let mut unordered = rows;
        unordered.swap(1, 2);
        assert!(means(&unordered, 1, 3).is_err());
    }
}

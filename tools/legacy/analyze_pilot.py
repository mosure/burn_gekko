#!/usr/bin/env python3
"""Summarize recorded pilot evidence and export figures; never trains or selects a model.

Requires matplotlib, numpy, safetensors and zstandard in an analysis environment.
"""
import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(study, prefix):
    runs = {}
    for name in ["b1-online", "b4-online", "b4-cache", "overfit", "smallset", "resume-cache"]:
        report = read(Path(".data/runs") / f"{prefix}-{name}" / "pilot-report.json")
        warm = report["steps"][report["warmup_steps_excluded"]:]
        telemetry_name = "resume-cache" if name == "resume-cache" else name
        telemetry = [json.loads(line) for line in (study / "execution" / f"{telemetry_name}-gpu.jsonl").read_text().splitlines()]
        steady_start = report["prepare_seconds"] + sum(s["elapsed_ms"] for s in report["steps"][:2]) / 1000
        steady = [r for r in telemetry if steady_start <= r["elapsed_seconds"] <= report["run_seconds"]]
        summary = {k: v for k, v in report.items() if k not in {"steps", "config", "probes"}}
        summary["phase_mean_ms"] = {
            k: statistics.mean(s[k] for s in warm) if warm[0][k] is not None else None
            for k in ["encoder_ms", "forward_loss_ms", "backward_clip_ms", "optimizer_ms"]
        }
        summary["peak_process_vram_mib"] = max((r.get("process_vram_mib", 0) for r in telemetry), default=0)
        summary["peak_process_rss_mib"] = max((r.get("process_rss_mib", 0) for r in telemetry), default=0)
        summary["steady_telemetry_samples"] = len(steady)
        summary["steady_device_gpu_percent_mean"] = statistics.mean(r["device_gpu_percent"] for r in steady) if steady else None
        summary["steady_device_power_w_mean"] = statistics.mean(r["device_power_w"] for r in steady) if steady else None
        summary["initial_probe"] = report["probes"][0]
        summary["final_probe"] = report["probes"][-1]
        summary["optimizer_examples"] = len(report["steps"]) * report["config"]["batch_size"]
        summary["unique_train_room_targets"] = len({tuple(s) for row in report["steps"] for s in row["samples"]})
        norms = [s["gradient_norm"] for s in report["steps"]]
        summary["gradient_norm"] = dict(min=min(norms), median=statistics.median(norms), max=max(norms), clipped_fraction=sum(n > report["config"]["training"]["global_grad_clip"] for n in norms) / len(norms))
        runs[name] = summary
    original = read(Path(".data/runs") / f"{prefix}-b4-cache/pilot-report.json")
    resumed = read(Path(".data/runs") / f"{prefix}-resume-cache/pilot-report.json")
    paired = list(zip(original["steps"][4:], resumed["steps"]))
    resume = {
        "max_loss_abs_difference": max(abs(a[k]-b[k]) for a,b in paired for k in ["total", "cross", "mae", "ri"]),
        "max_gradient_norm_abs_difference": max(abs(a["gradient_norm"]-b["gradient_norm"]) for a,b in paired),
        "same_samples": all(a["samples"] == b["samples"] for a,b in paired),
        "max_final_probe_abs_difference": max(abs(original["probes"][-1][split][key]-resumed["probes"][-1][split][key]) for split in ["train", "validation"] for key in ["total", "cross", "mae", "ri"]),
    }
    assert resume["same_samples"] and resume["max_loss_abs_difference"] < 1e-5
    evaluations = {}
    for split in ["validation", "test"]:
        evaluations[split] = {}
        for step in [0, 128]:
            report = read(Path(".data/runs") / f"{prefix}-smallset/step-{step:06}-evaluation-{split}.json")
            report["random_ranking_ap_baseline"] = report["covisibility"]["positives"] / report["covisibility"]["pixels"]
            evaluations[split][str(step)] = report
    geometry = read(study / "geometry-audit.json")
    pairs = geometry.pop("pairs")
    geometry["directed_pairs"] = len(pairs)
    geometry["split_visible_fraction"] = {}
    for split in ["train", "validation", "test"]:
        fractions = [p["visible"] / sum(p[k] for k in ["visible", "occluded", "out_of_view", "unknown"]) for p in pairs if p["split"] == split]
        geometry["split_visible_fraction"][split] = dict(min=min(fractions), median=statistics.median(fractions), max=max(fractions))
    ledgers = [read(study / d / "ledger.json") for d in ["execution", "evaluation"]]
    summary = dict(
        schema=1, diagnostic_only=True, runs=runs, cuda_resume=resume,
        evaluations=evaluations, geometry=geometry,
        command_seconds=sum(x["command_seconds"] for x in ledgers),
        completed_optimizer_steps=sum(x["completed_steps"] for n,x in runs.items() if n != "resume-cache") + 4,
        caveats=["One seed and two rooms per held-out split; pixels are correlated.", "Warm timing excludes two steps and preparation; integer millisecond timings are approximate.", "Eight-step performance screens run sequentially with different cold-cache histories; not a saturated-GPU benchmark.", "GPU telemetry is device-wide at 1 Hz and includes desktop activity; steady sample counts are small.", "Frozen full features are resident per run; masked targets are re-encoded every step.", "Local encoder package verified; independent original-checkpoint numerical parity remains open."],
    )
    (study / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def plot(study, prefix, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    names = ["b1-online", "b4-online", "b4-cache"]
    values = [summary["runs"][n]["warm_examples_per_second"] for n in names]
    axes[0, 0].bar(["B1 online", "B4 online", "B4 cached"], values, color=["#94a3b8", "#3b82f6", "#0d9488"])
    for i,v in enumerate(values):
        axes[0, 0].text(i, v+3, f"{v:.1f}", ha="center")
    axes[0, 0].set(title="Short throughput screens (6 measured steps)", ylabel="Target examples / second", ylim=(0,max(values)*1.2))
    for i,name in enumerate(["overfit", "smallset"]):
        report = read(Path(".data/runs") / f"{prefix}-{name}/pilot-report.json")
        ax = axes[0, 1] if i == 0 else axes[1, 0]
        for split, color in [("train", "#2563eb"), ("validation", "#ea580c")]:
            ax.plot([p["step"] for p in report["probes"]], [p[split]["total"] for p in report["probes"]], marker="o", label=split, color=color)
        ax.set(title="Fixed-example fit" if i == 0 else "12-room training: fixed four-example probes", xlabel="Optimizer step", ylabel="Total loss")
        ax.legend()
    for split,color in [("validation", "#ea580c"), ("test", "#7c3aed")]:
        data = summary["evaluations"][split]
        axes[1, 1].plot([0,128], [data[s]["covisibility"]["auroc"] for s in ["0","128"]], marker="o", label=split, color=color)
    axes[1, 1].axhline(.5, color="#64748b", linestyle="--", label="chance")
    axes[1, 1].set(title="Held-out co-visibility remains near chance", xlabel="Optimizer step", ylabel="Pixel AUROC", ylim=(.45,.55))
    axes[1, 1].legend()
    fig.suptitle("V-JEPA 2.1 + Gekko pilot 01 · RTX PRO 6000 · F32 · one seed", fontsize=15)
    fig.savefig(study / "pilot-summary.svg")
    fig.savefig(study / "pilot-summary.png", dpi=160)
    plt.close(fig)
    # Qualitative capture inspection only; no reconstruction targets or geometry used by training.
    import zstandard
    from safetensors.numpy import load
    protocol = read(study / "protocol.json")
    dataset = Path(protocol["plan"]["commands"][0]["argv"][3])
    manifest = read(dataset / "manifest.json")
    entries = [manifest["scenes"][i] for i in [0,1,12,14]]
    fig, axes = plt.subplots(4, 3, figsize=(9, 11), layout="constrained")
    for row, entry in enumerate(entries):
        data = (dataset / "raw" / entry["file"]).read_bytes()
        tensors = load(zstandard.ZstdDecompressor().decompress(data, max_output_size=64*1024*1024))
        rgb = tensors["color"][0,0]
        for view in range(3):
            axes[row,view].imshow(np.clip(rgb[view],0,1))
            axes[row,view].set_title(f"{entry['split']} {entry['seed']} · view {view}", fontsize=9)
            axes[row,view].axis("off")
    fig.suptitle("Published Zeroverse 0.22.0 · connected three-view room captures")
    fig.savefig(study / "capture-samples.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(".data/pilot-01"))
    parser.add_argument("--prefix", default="pilot-01")
    args = parser.parse_args()
    if not args.study.resolve().is_relative_to(Path(".data").resolve()):
        parser.error("study outputs must stay inside .data")
    summary = summarize(args.study, args.prefix)
    plot(args.study, args.prefix, summary)
    print(args.study / "summary.json")


if __name__ == "__main__":
    main()

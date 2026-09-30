"""Read immutable experiment artifacts and compute report statistics (no model selection)."""
import collections
import json
from pathlib import Path
import struct

import numpy as np
import zstandard


def read(path):
    return json.loads(Path(path).read_text())


def bootstrap(values, seed=2902, repeats=2000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "low": None, "high": None, "rooms": 0}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (repeats, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), low=float(np.quantile(means, .025)), high=float(np.quantile(means, .975)), rooms=len(values))


def room_metrics(report):
    rooms = collections.defaultdict(list)
    for target in report["targets"]:
        rooms[target["room_seed"]].append(target)
    result = {}
    for key in ["cross_mse", "mae_mse", "total_loss", "ri_loss", "auroc", "average_precision"]:
        values = {}
        for seed, targets in rooms.items():
            raw = [t["covisibility"].get(key) if key in {"auroc", "average_precision"} else t[key] for t in targets]
            raw = [v for v in raw if v is not None]
            if raw:
                values[seed] = float(np.mean(raw))
        result[key] = dict(per_room=values, bootstrap=bootstrap(list(values.values())))
    return result


def dataset_inventory(dataset, output):
    """Read generator manifests from a streaming zstd reader, without keeping image tensors."""
    manifest = read(dataset / "manifest.json")
    rows = []
    for scene in manifest["scenes"]:
        with (dataset / "raw" / scene["file"]).open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(length))
                start, end = header["indoor_manifest_0"]["data_offsets"]
                remaining = start
                while remaining:
                    chunk = stream.read(min(remaining, 1024*1024))
                    if not chunk:
                        raise ValueError("truncated capture")
                    remaining -= len(chunk)
                room = json.loads(stream.read(end-start))
        rows.append(dict(seed=scene["seed"], split=scene["split"], layout=room.get("layout"), palette=room.get("palette"), lighting=room.get("lighting"), target_lux=room.get("target_lux"), objects=len(room.get("objects", [])), humans=len(room.get("humans", [])), bytes=(dataset/"raw"/scene["file"]).stat().st_size))
    result = dict(dataset_id=manifest["dataset_id"], config=manifest["config"], rooms=rows, bytes=sum(r["bytes"] for r in rows))
    output.write_text(json.dumps(result, indent=2)+"\n")
    return result


def zero_baseline(dataset, sample_directory, output):
    """Post-hoc CPU diagnostic, using the exact saved evaluation mask and RGB contract."""
    meta=read(Path(sample_directory)/"sample.json")
    assert meta["normalize_targets"], "zero baseline expects patch-normalized RGB"
    manifest=read(dataset/"manifest.json")
    p=meta["patch_size"]; h,w=meta["height"],meta["width"]
    hidden=np.ones((h//p)*(w//p),dtype=bool)
    hidden[meta["visible_patch_ids"]]=False
    rows=[]
    for scene in manifest["scenes"]:
        if scene["split"]=="train":continue
        with (dataset/"raw"/scene["file"]).open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                length=struct.unpack("<Q",stream.read(8))[0]
                header=json.loads(stream.read(length)); desc=header["color"]
                assert desc["dtype"]=="F32" and desc["shape"]==[1,1,3,h,w,3]
                start,end=desc["data_offsets"]
                stream.read(start)
                rgb=np.frombuffer(stream.read(end-start),dtype="<f4").reshape(3,h,w,3)
        for view,image in enumerate(rgb):
            patches=image.reshape(h//p,p,w//p,p,3).transpose(0,2,1,3,4).reshape(-1,p*p*3)
            centered=patches-patches.mean(axis=1,keepdims=True)
            normalized=centered/np.sqrt(patches.var(axis=1,ddof=1,keepdims=True)+1e-6)
            spatial_contrast=np.sqrt(patches.reshape(-1,p*p,3).var(axis=1).mean(axis=1))
            rows.append(dict(room_seed=scene["seed"],split=scene["split"],target_view=view,
                zero_normalized_mse=float((normalized[hidden]**2).mean()),
                low_spatial_contrast_fraction=float((spatial_contrast<.01).mean())))
    result=dict(analysis="post-hoc diagnostic; zero prediction in patch-normalized RGB; exact fixed evaluation mask",
                low_contrast_definition="sqrt(mean over RGB channels of within-patch spatial variance) < 0.01 sRGB units; all patches",targets=rows)
    for split in ["validation","test"]:
        selected=[r for r in rows if r["split"]==split]
        result[split]={key:float(np.mean([r[key] for r in selected])) for key in ["zero_normalized_mse","low_spatial_contrast_fraction"]}
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result


def collect(study, run, dataset):
    training = read(run / "pilot-report.json")
    evals = {
        "validation_initial": read(run / "step-000000-evaluation-validation.json"),
        "validation_final": read(run / "evaluation-validation.json"),
        "test_initial": read(run / "step-000000-evaluation-test.json"),
        "test_final": read(run / "evaluation-test.json"),
        "test_unrelated": read(run / "unrelated-evaluation-test.json"),
    }
    for report in evals.values():
        report["room_metrics"] = room_metrics(report)
        metric = report["covisibility"]
        report["ap_prevalence_baseline"] = metric["positives"] / metric["pixels"]
    correct = {t["room_seed"]: [] for t in evals["test_final"]["targets"]}
    wrong = {(t["room_seed"],t["target_view"]):t for t in evals["test_unrelated"]["targets"]}
    mono_gain = collections.defaultdict(list)
    mae_differences = []
    for t in evals["test_final"]["targets"]:
        other = wrong[(t["room_seed"],t["target_view"])]
        correct[t["room_seed"]].append(other["cross_mse"] - t["cross_mse"])
        mono_gain[t["room_seed"]].append(t["mae_mse"] - t["cross_mse"])
        mae_differences.append(abs(t["mae_mse"]-other["mae_mse"]))
    control = dict(
        unrelated_minus_correct_mse=bootstrap([np.mean(v) for v in correct.values()]),
        mae_minus_cross_mse=bootstrap([np.mean(v) for v in mono_gain.values()]),
        max_mae_control_difference=max(mae_differences),
    )
    ledger_paths = sorted(study.glob("*/ledger.json"))
    ledgers = [read(p) for p in ledger_paths]
    commands = [dict(c, stage=p.parent.name) for p,l in zip(ledger_paths,ledgers) for c in l["commands"]]
    command_seconds = sum(l.get("command_seconds", 0) for l in ledgers)
    telemetry = [json.loads(line) for line in (study / "training/main-gpu.jsonl").read_text().splitlines()]
    elapsed = np.array([s.get("elapsed_ms_precise",s["elapsed_ms"]) for s in training["steps"]])
    warm = elapsed[min(10,len(elapsed)-1):]
    warm_start = training["prepare_seconds"] + float(elapsed[:10].sum()/1000)
    steady = [r for r in telemetry if warm_start <= r["elapsed_seconds"] <= training["run_seconds"]]
    performance = dict(
        warm_step_median_ms=float(np.median(warm)), warm_step_p90_ms=float(np.quantile(warm,.9)),
        warm_examples_per_second=float(len(warm)*training["config"]["batch_size"]/(warm.sum()/1000)),
        end_to_end_examples_per_second=len(training["steps"])*training["config"]["batch_size"]/training["run_seconds"],
        peak_process_vram_mib=max(r.get("process_vram_mib",0) for r in telemetry),
        steady_samples=len(steady), steady_device_gpu_percent_mean=float(np.mean([r["device_gpu_percent"] for r in steady])) if steady else None,
        steady_device_power_w_mean=float(np.mean([r["device_power_w"] for r in steady])) if steady else None,
        clipped_fraction=float(np.mean([s["gradient_norm"] > training["config"]["training"]["global_grad_clip"] for s in training["steps"]])),
    )
    window=min(5000,len(training["steps"])//2)
    performance["late_window_steps"]=window
    performance["late_training_windows"]={
        key:{"previous":float(np.mean([s[key] for s in training["steps"][-2*window:-window]])),
             "terminal":float(np.mean([s[key] for s in training["steps"][-window:]]))}
        for key in ["total","cross","mae","ri"]}
    inventory_path = study/"dataset-inventory.json"
    inventory = read(inventory_path) if inventory_path.exists() else dataset_inventory(dataset, inventory_path)
    geometry = read(study/"geometry-audit.json")
    zero=zero_baseline(dataset,evals["test_final"]["samples"][0],study/"zero-baseline.json")
    summary = dict(schema=1, study="pilot-02", training=training, evaluations=evals, reference_control=control,
        performance=performance, inventory=inventory, geometry=geometry, command_seconds=command_seconds,
        commands=commands, completed_steps=training["completed_steps"], dataset=str(dataset), run=str(run),
        sizing=read(study/"sizing.json"), machine=read(study/"machine-source.json"), zero_baseline=zero,
        resource_events=read(study/"resource-events.json") if (study/"resource-events.json").exists() else [],
        qualification="single-seed synthetic-room pilot; independent encoder parity and real-world transfer unqualified")
    (study/"report-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary, telemetry


def sample_arrays(directory):
    directory = Path(directory)
    meta = read(directory/"sample.json")
    h,w,p = meta["height"],meta["width"],meta["patch_size"]
    target = np.fromfile(directory/"target.f32",dtype="<f4").reshape(h,w,3)
    refs = [np.fromfile(directory/f"reference-{i}.f32",dtype="<f4").reshape(h,w,3) for i in range(len(meta["reference_views"]))]
    patches = target.reshape(h//p,p,w//p,p,3).transpose(0,2,1,3,4).reshape(-1,p*p*3)
    mean = patches.mean(axis=1,keepdims=True)
    std = np.sqrt(patches.var(axis=1,ddof=1,keepdims=True)+1e-6)
    normalized = (patches-mean)/std if meta["normalize_targets"] else patches
    def unpatch(array, channels):
        return array.reshape(h//p,w//p,p,p,channels).transpose(0,2,1,3,4).reshape(h,w,channels)
    outputs = {}
    for name in ["cross","mae"]:
        raw = np.fromfile(directory/f"{name}.f32",dtype="<f4").reshape(-1,p*p*3)
        if meta.get("predicted_patch_statistics",False):
            statistics=np.fromfile(directory/f"{name}-statistics.f32",dtype="<f4").reshape(-1,2)
            display=raw*np.exp(statistics[:,1:2].clip(-10,1))+statistics[:,0:1]
        else:
            display=raw*std+mean if meta["normalize_targets"] else raw
        outputs[name] = np.clip(unpatch(display,3),0,1)
        outputs[name+"_error"] = unpatch(((raw-normalized)**2).reshape(-1,p*p,3).mean(axis=2),1)[...,0]
    ri = unpatch(np.fromfile(directory/"ri.f32",dtype="<f4"),1)[...,0]
    visible = np.zeros((h//p)*(w//p),dtype=bool); visible[meta["visible_patch_ids"]]=True
    visible = visible.reshape(h//p,w//p).repeat(p,axis=0).repeat(p,axis=1)
    masked = target.copy(); masked[~visible] = [0.23,0.28,0.34]
    truth = np.fromfile(directory/"visibility.u8",dtype=np.uint8).reshape(h,w)
    empirical_ri = (outputs["mae_error"]-outputs["cross_error"])/np.maximum(outputs["mae_error"],.01)
    return dict(meta=meta,target=target,references=refs,masked=masked,hidden=~visible,ri=ri,visibility=truth,empirical_ri=empirical_ri,**outputs)

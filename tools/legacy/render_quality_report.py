#!/usr/bin/env python3
"""Render a reproducible reconstruction-quality study from a TOML artifact index."""
import argparse
import collections
import json
from pathlib import Path
import textwrap
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from reconstruction_diagnostics import inspect_sample
from pilot_report_data import bootstrap, room_metrics


def read(path):
    return json.loads(Path(path).read_text())


def run_record(spec):
    root=Path(spec["path"])
    train=read(root/"pilot-report.json")
    evaluation=read(root/spec.get("evaluation","evaluation-validation.json"))
    samples=[inspect_sample(p)[1] for p in evaluation["samples"]]
    times=np.array([s["elapsed_ms_precise"] for s in train["steps"]])[10:]
    return dict(spec=spec,training=train,evaluation=evaluation,samples=samples,
        throughput=float(len(times)*train["config"]["batch_size"]/(times.sum()/1000)),
        psnr=float(-10*np.log10(evaluation["mean_masked_rgb_mse"])),
        mean_sample_edge_cosine=float(np.mean([s["edge_cosine"] for s in samples])))


def paired_control(correct, control, compare_mae=True):
    other={(t["room_seed"],t["target_view"]):t for t in control["targets"]}
    diffs=collections.defaultdict(list)
    mae=[]
    for t in correct["targets"]:
        c=other[(t["room_seed"],t["target_view"])]
        diffs[t["room_seed"]].append(c["masked_rgb_mse"]-t["masked_rgb_mse"])
        if compare_mae:mae.append(abs(c["mae_mse"]-t["mae_mse"]))
    return dict(rgb_mse_increase=bootstrap([np.mean(v) for v in diffs.values()]),
                max_mae_difference=max(mae) if compare_mae else None)


def performance(study, spec, training):
    result={}
    for ledger_path in study.glob("*/ledger.json"):
        ledger=read(ledger_path)
        for command in ledger["commands"]:
            argv=command["argv"]
            if "train-pilot" not in argv or "--run" not in argv:
                continue
            if argv[argv.index("--run")+1] != Path(spec["path"]).name:
                continue
            rows=[json.loads(line) for line in (ledger_path.parent/(command["name"]+"-gpu.jsonl")).read_text().splitlines()]
            cold=training["prepare_seconds"]+sum(s["elapsed_ms_precise"] for s in training["steps"][:10])/1000
            steady=[r for r in rows if cold<=r["elapsed_seconds"]<=training["run_seconds"]]
            result=dict(command_seconds=command["elapsed_seconds"],prepare_seconds=training["prepare_seconds"],
                peak_process_vram_mib=max(r.get("process_vram_mib",0) for r in rows),
                end_to_end_examples_per_second=training["completed_steps"]*training["config"]["batch_size"]/command["elapsed_seconds"],
                warm_device_gpu_percent=float(np.mean([r["device_gpu_percent"] for r in steady])) if steady else None,
                warm_device_power_w=float(np.mean([r["device_power_w"] for r in steady])) if steady else None)
    return result


class Pages:
    def __init__(self,path):
        self.pdf=PdfPages(path);self.number=0
    def new(self,title,subtitle=""):
        f=plt.figure(figsize=(11.69,8.27),facecolor="white")
        f.text(.055,.95,"BURN GEKKO  /  PILOT 03",color="#0d9488",weight="bold",fontsize=10)
        f.text(.055,.90,title,color="#0f172a",weight="bold",fontsize=21)
        f.text(.055,.86,subtitle,color="#64748b",fontsize=9)
        return f
    def text(self,f,text,y,width=135):
        lines=textwrap.wrap(text,width=width,break_long_words=False)
        f.text(.06,y,"\n".join(lines),va="top",fontsize=10,linespacing=1.5,color="#334155")
        return y-.025*len(lines)-.035
    def save(self,f):
        self.number+=1
        f.text(.055,.025,"Single workstation • 256×256 procedural rooms • independent room splits",fontsize=8,color="#64748b")
        f.text(.95,.025,str(self.number),ha="right",fontsize=8,color="#64748b")
        f.canvas.draw()
        renderer=f.canvas.get_renderer()
        # Axis locators retain out-of-range tick labels which are not drawn.
        # Check rendered text instead of treating those hidden ticks as overflow.
        texts=list(f.texts)
        for ax in f.axes:
            texts.append(ax.title)
            if ax.axison:
                texts.extend([ax.xaxis.label,ax.yaxis.label])
                for axis in [ax.xaxis,ax.yaxis]:
                    for tick in axis._update_ticks():
                        texts.extend([tick.label1,tick.label2])
            for table in ax.tables:
                texts.extend(cell.get_text() for cell in table.get_celld().values())
            legend=ax.get_legend()
            if legend is not None:texts.extend(legend.get_texts())
        for text in texts:
            if text.get_visible() and text.get_text():
                box=text.get_window_extent(renderer)
                if box.x0 < -1 or box.y0 < -1 or box.x1 > f.bbox.width+1 or box.y1 > f.bbox.height+1:
                    raise ValueError(f"text extends outside PDF page {self.number}: {text.get_text()[:100]}")
        # Preserve at least the native 256-pixel sample resolution in compact panels.
        self.pdf.savefig(f,dpi=240);plt.close(f)
    def close(self):
        self.pdf.infodict().update(Title="Burn Gekko pilot 03: reconstruction diagnosis and controlled experiments")
        self.pdf.close()


def image_grid(f, rows, labels, rect=(.055,.13,.9,.65)):
    left,bottom,width,height=rect
    for r,images in enumerate(rows):
        for c,im in enumerate(images):
            ax=f.add_axes([left+c*width/len(images),bottom+(len(rows)-1-r)*height/len(rows),width/len(images)-.012,height/len(rows)-.045])
            ax.imshow(im,interpolation="nearest");ax.axis("off")
            if labels and isinstance(labels[0],list):ax.set_title(labels[r][c],fontsize=9)
            elif r==0:ax.set_title(labels[c],fontsize=9)


def build(config):
    study=Path(config["study"])
    runs=[run_record(r) for r in config["runs"]]
    for r in runs:r["performance"]=performance(study,r["spec"],r["training"])
    selected=next(r for r in runs if r["spec"]["name"]==config["selected"])
    final=read(Path(selected["spec"]["path"])/"evaluation-test.json")
    unrelated=read(Path(selected["spec"]["path"])/"unrelated-evaluation-test.json")
    control=paired_control(final,unrelated)
    ledgers=[read(p) for p in sorted(study.glob("*/ledger.json"))]
    seconds=sum(l.get("command_seconds",0) for l in ledgers)
    baseline=next(r for r in runs if r["spec"]["name"]==config["baseline"])
    baseline_test=read(Path(baseline["spec"]["path"])/"evaluation-test.json")
    baseline_control=paired_control(final,baseline_test,compare_mae=False)
    test_samples=[inspect_sample(p)[1] for p in final['samples']]
    mono_rgb=bootstrap([r['mae_hidden_rgb_mse']-r['hidden_rgb_mse'] for r in test_samples])
    test_room_metrics=room_metrics(final)
    summary=dict(runs=runs,selected=config["selected"],test=final,unrelated=unrelated,baseline_test=baseline_test,
                 baseline_control=baseline_control,
                 control=control,test_room_metrics=test_room_metrics,test_samples=test_samples,
                 exported_monocular_rgb_control=mono_rgb,command_seconds=seconds,findings=config["findings"])
    (study/"quality-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    pages=Pages(study/"burn_gekko_pilot_03_report.pdf")
    f=pages.new("Reconstruction quality: diagnosis and experiments",config["status"])
    y=.80
    for note in config["findings"]:y=pages.text(f,note,y)
    y=pages.text(f,f"Measured capture, audit, training and evaluation command time: {seconds/60:.1f} minutes; ceiling 120 minutes. Setup/build/report time is separate. Recipe decisions used validation rooms. Test model evaluation occurred after recipes and terminal checkpoints were fixed.",y)
    pages.save(f)

    f=pages.new("Objectives and visible-only control", "Separate content fidelity, RGB calibration, and co-visibility evidence")
    y=.80
    smooth=read(study/"calibrated-visible-baseline.json")
    for note in [
        "The normalized-content objective retains Gekko's masked RGB reconstruction terms and detached relative-improvement target. Raw RGB changes the scale of both reconstruction errors and the RI term; in the raw-only screens, RI loss was tiny and AUROC remained near 0.5. Lower raw RGB error did not establish useful co-visibility.",
        "The calibrated candidate predicts 768 normalized RGB values plus a patch mean and log standard deviation. For each RGB head, hidden-patch calibration adds 16 times squared mean error, 0.1 times squared log-standard-deviation error, and reconstructed RGB MSE. RI continues to use normalized content and receives no gradient through its RGB-error targets.",
        "These extra terms are included in total loss but not in the individual content/RI columns. Total loss is therefore not directly comparable across recipes. Calibrated inference uses only predicted patch statistics; ground-truth patch statistics appear in training targets, never in its inference RGB conversion.",
        f"A fixed smooth control interpolates observed target patch means using inverse squared spatial distance, with no training or references. On the 16 screen validation exports it scores {smooth['baseline_hidden_rgb_mse']:.6f} hidden RGB MSE; the calibrated screen scores {smooth['model_hidden_rgb_mse']:.6f} on those same exports. A hidden-pixel intervention leaves every control prediction unchanged.",
        "A model beating this smooth control still needs sharp, coherent edges and useful cross-view matching. The geometry warp, visible-only control, monocular branch and unrelated-reference intervention answer different questions and are kept separate."]:y=pages.text(f,note,y)
    pages.save(f)

    inventory=read(study/"dataset-inventory.json")
    f=pages.new("Data and isolation",f"Main cache: {inventory['dataset_id'][:24]}... • {inventory['bytes']/1e9:.2f} GB compressed")
    cells=[]
    for split in ['train','validation','test']:
        rows=[r for r in inventory['rooms'] if r['split']==split]
        cells.append([split,str(len(rows)),str(min(r['seed'] for r in rows)),str(max(r['seed'] for r in rows)),str(len(set(r['layout'] for r in rows)))])
    ax=f.add_axes([.08,.57,.84,.23]);ax.axis('off')
    tb=ax.table(cellText=cells,colLabels=['Split','Rooms','First seed','Last seed','Layouts'],loc='upper center',cellLoc='left')
    tb.auto_set_font_size(False);tb.set_fontsize(10);tb.scale(1,2)
    y=.48
    for note in [
        "The main cache contains 512 training, 32 validation and 32 test rooms, each with three 256-pixel views. Room seeds are disjoint across splits and from the earlier pilot-02 cache. Its 1,536 training target/reference tuples receive 144,000 presentations per arm (93.75 shuffled passes).",
        "The five screening recipes used 128/16/16 rooms from a separate cache. Some screening seeds overlap the historical pilot-02 training seeds, regenerated with the updated packages; all screen decoders start fresh. The main comparison uses a new seed range.",
        "Capture is validated before atomic cache publication. RGB, geometry, hashes, dimensions and split identities are checked. Room layouts are not stratified: the 32-room test split covers nine of the ten training layout families. This small single-seed sample limits uncertainty and transfer claims.",
        "Validation probes use a fixed 16-example panel. Final validation and test metrics each cover every target view in their 32 rooms (96 views), with one fixed mask. Geometry only supplies evaluation labels and a separately marked diagnostic warp."]:y=pages.text(f,note,y)
    pages.save(f)

    f=pages.new("What was verified", "Package provenance, encoder numerics and image reconstruction are separate checks")
    y=.80
    facts=[
      "Published generator versions: bevy_zeroverse 0.23.0 and bevy_zeroverse_burn 0.6.0. Registry checksums matched both downloaded archives. Capture identities changed; historical caches and reports remain separate.",
      "All 158 imported encoder tensors exactly match Meta's official EMA checkpoint after F16 conversion. Native-image dense and sparse CPU outputs match official PyTorch using the same quantized weights expanded to F32 at 32 and 256 pixels (relative RMS error below 0.000004). Error versus the original F32 weights is at most 0.000773.",
      "CUDA outputs differ by up to 0.0029 relative RMS at 384 pixels, cosine above 0.999995. They failed the original 0.001 RMS gate. F32 storage may use TF32 accelerated arithmetic; the GPU residual is recorded rather than called exact parity.",
      "A fixed training image/mask fits nearly exactly. This tests the optimizer, decoder and patch layout, but does not measure unseen-room quality or prove learned geometric correspondence.",
      "Image metrics are recomputed independently from exported float arrays. Hidden RGB MSE uses unclipped predictions. Visualizations only clip to [0,1] and copy visible input patches. No image sharpening or hidden-pixel compositing is applied."
    ]
    decoder_check=read(study/'decoder-backend-corrected-audit/comparison.json')
    residual=max(v['relative_rms'] for v in decoder_check['metrics'].values())
    facts.append(f"Decoder CPU/CUDA prediction and input-gradient check: {'passed' if decoder_check['passed'] else 'FAILED'} the 0.01 relative RMS gate; worst residual {residual:.6f}. Scope: width 384, two blocks, batch two, 4x4 token grid, rotary attention, corrected MAE and learned calibration. This is a backend check, not official Gekko parity.")
    for note in facts:y=pages.text(f,note,y)
    pages.save(f)

    f=pages.new("Can the reference images explain the target?", "Ground-truth geometry diagnostic only: geometry is never provided to training")
    ax=f.add_axes([.04,.19,.92,.63]);ax.imshow(plt.imread(study/"geometry-rgb/geometry-oracle.png"));ax.axis("off")
    pages.text(f,"Using ground-truth depth and camera transforms to warp reference RGB produces sharp aligned structure. On covered pixels, mean RGB MSE is 0.000481, versus 0.011748 for unwarped references. This confirms useful correspondence information in the data; it is not a model result, and its coverage differs from the model's hidden-pixel metric.",.15)
    pages.save(f)

    f=pages.new("Experiment design and qualification")
    y=.80
    for note in config["design"]:y=pages.text(f,note,y)
    pages.save(f)

    f=pages.new("Workstation efficiency", "RTX PRO 6000 Blackwell • frozen V-JEPA Base • CUDA F32 storage • no AMP")
    cells=[]
    for r in runs:
        p=r["performance"]
        cells.append([r["spec"]["label"],str(r["training"]["config"]["batch_size"]),f"{r['throughput']:.1f}",
            f"{p['end_to_end_examples_per_second']:.1f}",f"{p['peak_process_vram_mib']/1024:.1f}",
            f"{p['warm_device_gpu_percent']:.1f}",f"{p['command_seconds']/60:.1f}"])
    ax=f.add_axes([.055,.43,.90,.37]);ax.axis("off")
    tb=ax.table(cellText=cells,colLabels=["Variant","Batch","Warm ex/s","Total ex/s","Peak GiB","GPU %","Minutes"],loc="upper center",cellLoc="left",colWidths=[.25,.08,.13,.13,.13,.13,.15])
    tb.auto_set_font_size(False);tb.set_fontsize(9);tb.scale(1,1.8)
    pages.text(f,"Warm throughput excludes the first ten optimizer steps. Total throughput includes loading, feature caching, compilation, probes and checkpoint writes in the training command. GPU utilization is device-wide and includes desktop activity; VRAM is attributed to the training process. Full-view features are cached; masked targets are re-encoded each step.",.31)
    pages.text(f,config["sizing_note"],.18)
    pages.save(f)

    over=read(Path(config["overfit_run"])/"pilot-report.json")
    over_eval=read(Path(config["overfit_run"])/"evaluation-training-diagnostic.json")
    a,om=inspect_sample(over_eval["samples"][0])
    f=pages.new("Exact-image reconstruction check",f"Training image only • {over['completed_steps']:,} updates • hidden PSNR {om['hidden_rgb_psnr']:.2f} dB • edge cosine {om['edge_cosine']:.4f}")
    image_grid(f,[[a["target"],a["masked"],a["completion"]]], ["Target","Masked input","Reconstruction (oracle statistics)"],(.06,.30,.88,.49))
    pages.text(f,"This is a memorization diagnostic with the fixed training mask. The validation error remains high. Its value is showing that sharp detail can pass through the patch layout and RGB decoder when the mapping is learned.",.23)
    pages.save(f)

    for stage in dict.fromkeys(r["spec"]["stage"] for r in runs):
        group=[r for r in runs if r["spec"]["stage"]==stage]
        f=pages.new(f"{stage.capitalize()}: held-out validation", "Compare standalone rows with each other; oracle rows receive hidden target patch statistics")
        cells=[]
        for r in group:
            e=r["evaluation"];t=r["training"]
            cells.append([r["spec"]["label"],f"{t['completed_steps']:,}","oracle" if e["rgb_uses_target_statistics"] else "standalone",
                f"{e['mean_masked_rgb_mse']:.5f}",f"{r['psnr']:.2f}",f"{r['mean_sample_edge_cosine']:.3f}",f"{e['covisibility']['auroc']:.3f}",f"{r['throughput']:.1f}"])
        ax=f.add_axes([.045,.59,.91,.22]);ax.axis("off")
        tb=ax.table(cellText=cells,colLabels=["Variant","Steps","RGB display","Hidden MSE","PSNR dB","Edge cosine*","RI AUROC","Examples/s"],loc="upper center",cellLoc="left",colWidths=[.21,.07,.12,.11,.09,.10,.10,.10]);tb.auto_set_font_size(False);tb.set_fontsize(8);tb.scale(1,1.8)
        pages.text(f,"* Edge cosine is measured on the predetermined target-zero exports, restricting gradients to pairs of hidden pixels. MSE/AUROC cover all evaluated target views. These are different endpoints: a low MSE can coexist with blurred edges.",.50)
        for i,r in enumerate(group):
            ax=f.add_axes([.07+i*.90/len(group),.14,.90/len(group)-.055,.26])
            probes=r["training"]["probes"]
            for split in ["train","validation"]:ax.plot([p["step"] for p in probes],[p[split]["cross"] for p in probes],label=split)
            ax.set_yscale("log");ax.set_title(r["spec"]["label"],fontsize=9);ax.set_xlabel("Optimizer step",fontsize=9);ax.tick_params(labelsize=8);ax.legend(fontsize=7)
        pages.save(f)

    f=pages.new("Candidate convergence", "Training curves use consecutive 500-update means; validation is the fixed monitoring panel")
    steps=selected['training']['steps']
    chunks=[steps[i:i+500] for i in range(0,len(steps),500)]
    xx=[c[-1]['step'] for c in chunks]
    means={key:[np.mean([s[key] for s in c]) for c in chunks] for key in ['cross','mae','ri','total']}
    probes=selected['training']['probes'][1:]
    specs=[('Normalized content',['cross','mae']),('RI loss',['ri']),('Total objective',['total'])]
    for i,(title,keys) in enumerate(specs):
        ax=f.add_axes([.065+i*.305,.38,.25,.40])
        for key in keys:
            line=ax.plot(xx,means[key],label='train '+key)[0]
            ax.plot([p['step'] for p in probes],[p['validation'][key] for p in probes],'--o',markersize=3,color=line.get_color(),label='validation '+key)
        ax.set_title(title,fontsize=11);ax.set_xlabel('Optimizer step');ax.grid(alpha=.2);ax.legend(fontsize=7)
    y=.28
    for note in [
        'Full-view features stay frozen and cached; target masks change by step. Training window means average sampled rooms and masks. The validation probe is a different fixed 16-example panel, so the vertical train/validation gap alone does not measure overfitting.',
        f"Final 500-update means: cross {means['cross'][-1]:.5f}, MAE {means['mae'][-1]:.5f}, RI {means['ri'][-1]:.5f}, calibration auxiliary {means['total'][-1]-means['cross'][-1]-means['mae'][-1]-means['ri'][-1]:.5f}. This is a bounded learning curve, not proof of convergence or useful co-visibility."]:
        y=pages.text(f,note,y)
    pages.save(f)

    comparisons=[r for r in runs if r["spec"]["stage"]=="screen"]
    paths=selected["evaluation"]["samples"]
    # Screen visual comparison uses one shared sample set and deterministic room indices.
    if comparisons:
        sample_maps=[{read(Path(p)/"sample.json")["room_seed"]:p for p in r["evaluation"]["samples"]} for r in comparisons]
        seeds=sorted(set.intersection(*(set(m) for m in sample_maps)))
        chosen=[seeds[i*len(seeds)//4] for i in range(min(4,len(seeds)))]
        for start in range(0,len(chosen),2):
            rows=[]
            for seed in chosen[start:start+2]:
                data=[inspect_sample(m[seed])[0] for m in sample_maps]
                rows.append([data[0]["target"]]+[a["completion"] for a in data])
            f=pages.new("Matched validation completions", "Room seeds: "+", ".join(map(str,chosen[start:start+2]))+" • target view 0 • identical evaluation mask")
            labels=["Target"]+[r["spec"]["label"]+(" (oracle)" if r["evaluation"]["rgb_uses_target_statistics"] else "") for r in comparisons]
            image_grid(f,rows,labels)
            pages.save(f)

    comparisons=[r for r in runs if r["spec"]["stage"]=="main"]
    if comparisons:
        maps=[{read(Path(p)/"sample.json")["room_seed"]:p for p in r["evaluation"]["samples"]} for r in comparisons]
        seeds=sorted(set.intersection(*(set(m) for m in maps)))
        chosen=[seeds[i*len(seeds)//4] for i in range(min(4,len(seeds)))]
        f=pages.new("Main comparison: held-out rooms", "Predetermined room indices • identical target masks • visible input patches are composited in both models")
        rows=[]
        for seed in chosen:
            data=[inspect_sample(m[seed])[0] for m in maps]
            rows.append([data[0]["target"],data[0]["masked"]]+[a["completion"] for a in data])
        image_grid(f,rows,["Target","Masked input"]+[r["spec"]["label"] for r in comparisons],(.06,.09,.88,.73))
        pages.save(f)

    f=pages.new("Candidate: held-out test",f"{config['selected']} • {final['evaluated_target_views']} target views • {len(set(t['room_seed'] for t in final['targets']))} independent rooms")
    y=.80
    gain=1-final["mean_masked_rgb_mse"]/unrelated["mean_masked_rgb_mse"]
    for note in [
        f"Hidden RGB MSE {final['mean_masked_rgb_mse']:.6f}; PSNR {-10*np.log10(final['mean_masked_rgb_mse']):.2f} dB. Predicted RGB requires no hidden target means or variances.",
        f"Matched semantic-only baseline: RGB MSE {baseline_test['mean_masked_rgb_mse']:.6f}. Candidate reduction {100*(1-final['mean_masked_rgb_mse']/baseline_test['mean_masked_rgb_mse']):.2f}%. Paired room bootstrap for baseline-minus-candidate MSE: {baseline_control['rgb_mse_increase']['mean']:.6f}, 95% interval [{baseline_control['rgb_mse_increase']['low']:.6f}, {baseline_control['rgb_mse_increase']['high']:.6f}].",
        f"Cross-view versus monocular reconstruction: {100*final['relative_cross_improvement']:.2f}% lower content-objective MSE (patch-normalized for calibrated models). RI AUROC {final['covisibility']['auroc']:.4f}; AP {final['covisibility']['average_precision']:.4f}; visibility prevalence {final['covisibility']['positives']/final['covisibility']['pixels']:.4f}.",
        f"Standalone RGB on the {len(test_samples)} predetermined target-zero exports: monocular MSE {np.mean([r['mae_hidden_rgb_mse'] for r in test_samples]):.6f}, cross-view MSE {np.mean([r['hidden_rgb_mse'] for r in test_samples]):.6f}. Paired room interval for monocular-minus-cross RGB MSE: [{mono_rgb['low']:.6f}, {mono_rgb['high']:.6f}]. This subset result is separate from the all-view content metric.",
        f"Correct references versus unrelated rooms: {100*gain:.2f}% lower hidden RGB MSE. Paired room bootstrap for unrelated-minus-correct MSE: {control['rgb_mse_increase']['mean']:.6f}, 95% interval [{control['rgb_mse_increase']['low']:.6f}, {control['rgb_mse_increase']['high']:.6f}]. MAE control maximum difference {control['max_mae_difference']:.3g}.",
        "Reference sensitivity does not establish correct geometric matching. RI ranks reconstruction utility rather than directly predicting a calibrated co-visibility probability. Unknown geometric pixels are excluded from ranking metrics.",
        config["quality_assessment"]]:y=pages.text(f,note,y)
    pages.save(f)

    macro=test_room_metrics['auroc']['bootstrap']
    f=pages.new("Co-visibility and uncertainty", "Bootstrap unit: independent room, not millions of correlated pixels")
    y=.80
    for note in [
        f"Pooled pixel AUROC is {final['covisibility']['auroc']:.4f}. The mean of per-room, per-view AUROCs is {macro['mean']:.4f}, with room-bootstrap 95% interval [{macro['low']:.4f}, {macro['high']:.4f}] over {macro['rooms']} rooms with defined AUROC. These aggregation methods weight the observations differently.",
        f"Pooled average precision is {final['covisibility']['average_precision']:.4f}; the positive prevalence baseline is {final['covisibility']['positives']/final['covisibility']['pixels']:.4f}. A high AP can be uninformative when almost every pixel is visible in a reference.",
        "Intervals use 2,000 deterministic bootstrap resamples of room-level values. They describe variation among this small synthetic room sample, not uncertainty across optimization seeds, generator versions, real scenes or alternate masks.",
        "The original screen's amended co-visibility gate failed. Advancing its calibrated recipe for a larger reconstruction diagnostic was an explicit protocol deviation recorded before main training and test model evaluation. It must not be described as a fully qualified screen winner.",
        "No pretrained Gekko decoder, geometry supervision, camera input, or real-world evaluation was used. The learned reference bank is order-invariant, and still requires stronger evidence of correspondence and transfer."]:y=pages.text(f,note,y)
    pages.save(f)

    shown=[final["samples"][i*len(final["samples"])//4] for i in range(min(4,len(final["samples"])))]
    for path in shown:
        a,m=inspect_sample(path)
        f=pages.new(f"Test sample: room {m['room_seed']}",f"Predetermined target view 0 • hidden PSNR {m['hidden_rgb_psnr']:.2f} dB • edge cosine {m['edge_cosine']:.3f} • edge energy {100*m['predicted_to_true_edge_energy']:.1f}% of target")
        mono=a["mae"].copy();mono[~a["hidden"]]=a["target"][~a["hidden"]]
        labels=a["visibility"]
        visibility=np.full((*labels.shape,3),.65)
        visibility[labels==0]=[.85,.25,.25]
        visibility[labels==1]=[.15,.75,.40]
        error=np.abs(a["completion"]-a["target"]).mean(2)
        error_rgb=plt.get_cmap("magma")(np.clip(error/.2,0,1))[...,:3]
        ri_rgb=plt.get_cmap("coolwarm")((a["ri"].clip(-1,1)+1)/2)[...,:3]
        image_grid(f,[[a["target"],a["masked"],a["references"][0]],
                      [a["references"][1],a["completion"],mono],
                      [error_rgb,ri_rgb,visibility]],
                      [["Target","Input: 75% hidden","Reference 1"],
                       ["Reference 2","Cross-view completion","Monocular completion"],
                       ["Absolute RGB error","Predicted RI score","Geometric visibility"]],(.065,.19,.87,.64))
        pages.text(f,"Error: black = 0, pale yellow = 0.20 or greater. RI: blue = -1, white = 0, red = +1; an unconstrained utility score, not a visibility probability. Visibility: green = visible, red = negative, gray = unknown. Geometry is evaluation-only. The RI branch sees the full target; RGB completion receives only visible target patches.",.13)
        pages.save(f)

    f=pages.new("Reproduction and limits")
    y=.80
    for note in config["limitations"]:y=pages.text(f,note,y)
    for note in [f"Study artifacts: {study}. Machine-readable quality-summary.json includes full training/evaluation records. Command plans, cumulative ledgers and GPU telemetry are retained per stage.",
       "User-facing experiment inputs are TOML. Generated tensors, reports and ledgers are float arrays, JSON and JSONL. Exact runtime binaries and source snapshots are archived for the screen and final model variants.",
       "References: crates.io bevy_zeroverse 0.23.0 / bevy_zeroverse_burn 0.6.0; Meta V-JEPA source 204698b45b3712590f06245fbfba32d3be539812; Gekko source 63f0ec9957885ea82cc2f4637d502003fbf9afcb. Full URLs, checkpoint hashes and qualifications are in docs/source-audit.md."]:y=pages.text(f,note,y)
    pages.save(f);pages.close()
    print(study/"burn_gekko_pilot_03_report.pdf")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True)
    args=p.parse_args();build(tomllib.loads(args.config.read_text()))

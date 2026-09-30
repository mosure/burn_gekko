#!/usr/bin/env python3
"""Generate the pilot review PDF from recorded artifacts, with annotated held-out samples."""
import argparse
import collections
import datetime
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from pilot_report_data import collect, read, sample_arrays

BLUE, ORANGE, TEAL, GRAY = "#2563eb", "#ea580c", "#0d9488", "#64748b"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "pdf.fonttype":42,
    "axes.spines.top":False, "axes.spines.right":False, "axes.titlesize":11})


class Report:
    def __init__(self, path):
        self.path = path
        self.pdf = PdfPages(path)
        self.page = 0
    def new(self, title, subtitle=None, landscape=False):
        fig = plt.figure(figsize=(11.69,8.27) if landscape else (8.27,11.69), facecolor="white")
        fig.text(.07,.955,"BURN GEKKO  /  PILOT 02",color=TEAL,fontsize=9,weight="bold")
        fig.text(.07,.916,title,fontsize=20,weight="bold",color="#0f172a")
        if subtitle:
            fig.text(.07,.885,subtitle,color=GRAY,fontsize=9)
        return fig
    def save(self, fig):
        self.page += 1
        fig.text(.07,.025,"Single workstation • procedural rooms • frozen V-JEPA 2.1 Base • F32",color=GRAY,fontsize=7)
        fig.text(.93,.025,f"{self.page:02}",ha="right",color=GRAY,fontsize=8)
        self.pdf.savefig(fig)
        plt.close(fig)
    def close(self):
        self.pdf.infodict().update(Title="Burn Gekko pilot 02: training and evaluation report",Author="burn_gekko experiment pipeline",Subject="Bounded synthetic-room training with held-out evaluation and annotated samples")
        self.pdf.close()


def paragraph(fig, text, y, size=10, width=96, color="#334155"):
    lines = textwrap.wrap(text,width=width,break_long_words=False)
    fig.text(.07,y,"\n".join(lines),va="top",fontsize=size,color=color,linespacing=1.5)
    return y-len(lines)*.019-.024


def table(fig, rows, columns, rect, sizes=None, fontsize=9):
    ax=fig.add_axes(rect);ax.axis("off")
    obj=ax.table(cellText=rows,colLabels=columns,loc="upper left",cellLoc="left",colLoc="left",colWidths=sizes)
    obj.auto_set_font_size(False);obj.set_fontsize(fontsize);obj.scale(1,1.7)
    for (row,col),cell in obj.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        if row==0:cell.set_facecolor("#e2e8f0");cell.set_text_props(weight="bold")
        elif row%2:cell.set_facecolor("#f8fafc")
    return obj


def fmt_ci(metric, digits=4):
    if metric["mean"] is None:return "not defined"
    return f"{metric['mean']:.{digits}f} [{metric['low']:.{digits}f}, {metric['high']:.{digits}f}]"


def moving(values, window):
    return np.convolve(np.asarray(values),np.ones(window)/window,mode="valid")


def build_pdf(study, summary, telemetry):
    out=study/"burn_gekko_pilot_02_report.pdf"
    report=Report(out)
    train=summary["training"]; cfg=train["config"]; tcfg=cfg["training"]
    inv=summary["inventory"]; perf=summary["performance"]; ev=summary["evaluations"]
    val,test=ev["validation_final"],ev["test_final"]
    controls=summary["reference_control"]
    counts=collections.Counter(r["split"] for r in inv["rooms"])
    steps=train["steps"]; visits=len(steps)*cfg["batch_size"]
    epochs=visits/(counts["train"]*inv["config"]["cameras"])
    learned=controls["unrelated_minus_correct_mse"]["low"]>0

    fig=report.new("Training run for review",datetime.datetime.now(datetime.timezone.utc).strftime("Generated %Y-%m-%d %H:%M UTC • local artifacts, one seed"))
    y=.84
    y=paragraph(fig,f"Completed {train['completed_steps']:,} optimizer steps on {counts['train']:,} training rooms at {inv['config']['width']}×{inv['config']['height']}, with three independently encoded views per room. Total measured compute-job wall time was {summary['command_seconds']/60:.1f} minutes against the 120-minute ceiling. Stop reason: {train['stop_reason']}.",y)
    wrong_mse=ev["test_unrelated"]["mean_cross_mse"]
    mono_gain=100*(test["mean_mae_mse"]-test["mean_cross_mse"])/test["mean_mae_mse"]
    outcome=(f"Correct references reduce test cross MSE by {100*(wrong_mse-test['mean_cross_mse'])/wrong_mse:.1f}% versus unrelated rooms. Cross-view MSE is {abs(mono_gain):.1f}% {'below' if mono_gain>=0 else 'above'} the target-only branch." if learned else "A reliable reference-view benefit is not established by this pilot.")
    y=paragraph(fig,outcome,y,size=12,width=78,color=TEAL if learned else ORANGE)
    y=paragraph(fig,f"Final held-out pixel AUROC is {val['covisibility']['auroc']:.4f} on validation and {test['covisibility']['auroc']:.4f} on test. AP must be compared with the visible-pixel prevalence: {test['ap_prevalence_baseline']:.3f} on test. Reconstruction fitting, reference utility, and geometric co-visibility are separate outcomes.",y)
    rows=[
        ["Dataset",f"{counts['train']} / {counts['validation']} / {counts['test']} rooms (train / val / test)"],
        ["Training visits",f"{visits:,} target examples; {epochs:.1f} room/view passes"],
        ["Decoder",f"width {tcfg['decoder_width']}, {tcfg['decoder_depth']} layers, {tcfg['decoder_heads']} heads"],
        ["Warm throughput",f"{perf['warm_examples_per_second']:.1f} examples/s; {perf['warm_step_median_ms']:.1f} ms median step"],
        ["Peak process VRAM",f"{perf['peak_process_vram_mib']/1024:.2f} GiB"],
        ["Validation total loss",f"{ev['validation_initial']['mean_total_loss']:.4f} → {val['mean_total_loss']:.4f}"],
        ["Test total loss",f"{ev['test_initial']['mean_total_loss']:.4f} → {test['mean_total_loss']:.4f}"],
    ]
    table(fig,rows,["Measure","Recorded result"],[.06,y-.29,.88,.29],[.30,.70])
    paragraph(fig,"This is a single-seed synthetic-data pilot. The encoder package is checksum-verified, but independent official-checkpoint image parity and real-world transfer remain unqualified. No result here is a reproduced published Gekko benchmark.",.18,size=9,width=105)
    report.save(fig)

    fig=report.new("Resolution, data volume, and budget","Throughput determined the scale before the main run")
    sizing=summary["sizing"]
    rows=[]
    for resolution in ["256","384"]:
        screen=sizing["screens"][resolution]
        rows.append([resolution+" × "+resolution,f"{screen['settled_examples_per_second']:.1f}",f"{screen['settled_median_ms']:.1f}",f"{screen['peak_process_vram_mib']/1024:.2f}"])
    table(fig,rows,["Resolution","Examples/s","Median step (ms)","Peak VRAM (GiB)"],[.06,.67,.88,.17],[.25,.25,.25,.25])
    y=.66
    y=paragraph(fig,"Both screens used the same six generated room seeds, batch eight, a 256-wide/four-layer decoder, and 64 updates. Statistics use the last 32 updates of each screen, after cold compilation spikes. The 384-pixel screen reached 46.6% of the 256-pixel throughput, below the preset 65% threshold. The main run therefore uses 256×256.",y,size=9,width=106)
    y=paragraph(fig,f"Throughput sizing exceeded the registered cap of 2,048 training rooms at approximately 32 passes. We kept the cap, added 64 validation and 64 test rooms, and assigned the remaining training budget to more shuffled room/view visits. Actual endpoint exposure is {epochs:.1f} passes. Feature preparation, checkpoint/probe overhead, and held-out evaluation received separate reserves.",y,size=9,width=106)
    totals=collections.Counter()
    for c in summary["commands"]:totals[c["stage"]]+=c["elapsed_seconds"]
    rows=[[name,f"{seconds/60:.2f}"] for name,seconds in sorted(totals.items())]
    rows.append(["TOTAL",f"{summary['command_seconds']/60:.2f} / 120.00"])
    table(fig,rows,["Recorded command stage","Minutes"],[.06,.16,.88,.27],[.65,.35],8)
    paragraph(fig,"The failed 384-pixel post-capture check is included in the budget. Its complete rendered data were recovered after repairing the reader's old size limit; it was not rendered again. Command wall time includes idle CPU work inside each job, making it conservative relative to GPU-active time.",.13,size=8,width=120)
    report.save(fig)

    fig=report.new("Protocol and model","Preset budget, split isolation, and explicit experimental boundaries")
    y=.84
    paragraphs=[
        "The two-hour ceiling counts generation, CUDA profiling, feature preparation, training, and held-out evaluation command wall time. CPU compilation and report preparation are outside this compute budget. Resolution and data volume were selected from throughput screens, not test quality.",
        f"All rooms are generated by pinned published bevy_zeroverse 0.22.0 and bevy_zeroverse_burn 0.5.0, with static three-view connected camera rigs. Room seeds are disjoint across train, validation, and test. Test-room features are excluded from training and monitoring. The terminal budgeted checkpoint is the reported endpoint; there is no test-driven checkpoint selection.",
        f"Each RGB camera image passes independently through the same frozen V-JEPA 2.1 Base encoder. A {tcfg['mask_ratio']*100:.0f}% target mask removes patches before encoder attention. Full-view frozen features are cached in GPU memory; masked targets are encoded again each step. Target and reference-camera indices are not synthetic video timesteps.",
        "The CroCo-style decoder shares blocks across masked target-only reconstruction, reconstruction with an unordered reference set, and a full-target relative-improvement (RI) branch. RGB targets are normalized per 16×16 patch. Geometry is never an encoder input or training target.",
        "The loss is MAE pixel error + cross-view pixel error + squared RI residual. The RI target compares the two RGB errors with stopped gradients; its scale uses max(MAE error, 0.01). RI is an unconstrained utility score, not a calibrated visibility probability. This uses the released-code clamp variant.",
        f"AdamW uses peak LR {tcfg['learning_rate']}, weight decay {tcfg['weight_decay']}, and true global gradient clipping at {tcfg['global_grad_clip']}. The run uses linear warmup and cosine decay. Room/target pairs are shuffled reproducibly once per epoch; masks and optimizer schedule derive from absolute step indices.",
        f"Batch size is {cfg['batch_size']}. Fixed train and validation probes are computed every {cfg['eval_every']} steps with a fixed mask; they are small monitoring panels. Whole-split evaluation uses all three targets per held-out room with one separate fixed evaluation mask.",
    ]
    for p in paragraphs:y=paragraph(fig,p,y,size=9,width=108)
    report.save(fig)

    fig=report.new("What was trained","One shared decoder, three branches; the vision encoder stays frozen")
    ax=fig.add_axes([.06,.35,.88,.49]);ax.set(xlim=(0,10),ylim=(0,10));ax.axis("off")
    def box(x,y,w,h,label,color="#f1f5f9"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06,rounding_size=0.08",facecolor=color,edgecolor="#94a3b8"))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=9)
    def arrow(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=10,color=GRAY))
    box(.2,8.2,2.7,1.2,"Masked target RGB\n64 visible patch tokens")
    box(3.65,8.2,2.7,1.2,"Full target RGB\n256 patch tokens")
    box(7.1,8.2,2.7,1.2,"Two reference images\n512 patch tokens total")
    box(.2,5.9,9.6,1.1,"Frozen V-JEPA 2.1 Base, applied independently to each image", "#ccfbf1")
    for x in [1.55,5,8.45]:arrow(x,8.15,x,7.05)
    box(.2,3.3,2.7,1.3,"MAE branch\nmasked target only", "#ffedd5")
    box(3.65,3.3,2.7,1.3,"Cross-view branch\nmasked target + refs", "#dbeafe")
    box(7.1,3.3,2.7,1.3,"RI branch\nfull target + refs", "#ccfbf1")
    for x in [1.55,5,8.45]:arrow(x,5.85,x,4.65)
    ax.text(5,5.22,"All branches share 4 decoder blocks; width 256, 8 attention heads",ha="center",fontsize=8,color=GRAY)
    box(.2,.7,2.7,1.2,"MAE RGB prediction\nseparate linear head")
    box(3.65,.7,2.7,1.2,"Cross RGB prediction\nseparate linear head")
    box(7.1,.7,2.7,1.2,"Per-pixel RI score\nseparate linear head")
    for x in [1.55,5,8.45]:arrow(x,3.25,x,1.95)
    y=.29
    for p in ["Before attention, masked-target tokens are selected from the 16×16 patch grid. Full targets and references use cached dense features. The decoder restores learned mask tokens and adds 2D sine/cosine positions. References share a role embedding and are concatenated into one unordered attention context.","The full target enters only the RI branch. Both RGB reconstruction branches receive the sparse masked target. All three losses are measured on hidden target pixels; RGB errors used as RI targets are detached. Encoder parameters receive no gradients.","Depth, positions, cameras, and geometric visibility are reserved for data validation and evaluation. They never enter the RGB encoder, decoder inputs, or training loss. Report images restore normalized predictions with target patch statistics only for labeled visualization."]:
        y=paragraph(fig,p,y,size=9,width=106)
    report.save(fig)

    fig=report.new("Dataset and geometry audit","Rendered overlap is measured; the camera sampler uses a proxy constraint")
    geo=summary["geometry"]
    rows=[["Image size",f"{inv['config']['width']} × {inv['config']['height']}"],["Stored capture size",f"{inv['bytes']/1024**3:.2f} GiB"],["Valid source pixels",f"{geo['valid_source_pixels']:,}"],["Maximum reprojection error",f"{geo['max_self_reprojection_pixels']:.6f} pixels"],["Maximum depth error",f"{geo['max_self_depth_error']:.8f} m"],["Directed pairs audited",f"{len(geo['pairs']):,}"]]
    table(fig,rows,["Contract","Measurement"],[.06,.63,.88,.21],[.5,.5])
    ax=fig.add_axes([.11,.36,.82,.23])
    for split,color in [("train",BLUE),("validation",ORANGE),("test",TEAL)]:
        pairs=[p for p in geo["pairs"] if p["split"]==split]
        fraction=[p["visible"]/sum(p[k] for k in ["visible","occluded","out_of_view","unknown"]) for p in pairs]
        ax.hist(fraction,bins=np.linspace(0,1,21),density=True,histtype="step",linewidth=1.7,label=split,color=color)
    ax.set(xlabel="Visible fraction of source pixels, one directed pair",ylabel="Density",title="Actual overlap distribution");ax.legend()
    layouts=collections.Counter(str(r["layout"]) for r in inv["rooms"])
    y=paragraph(fig,"Layout counts: "+", ".join(f"{k}: {v}" for k,v in sorted(layouts.items())),.30,size=9,width=104)
    y=paragraph(fig,"The float32 position=2 export is an unclamped affine AABB coordinate. Values outside [0,1] are decoded to world coordinates. Camera-Z depth and column-major world-from-view matrices are verified by self-projection. Visibility uses nearest-pixel depth with max(0.02 m, 1% depth) tolerance; front-of-surface mismatches are unknown.",y,size=9,width=104)
    paragraph(fig,"Data are immutable, checksum-verified zstd Safetensors. The seed/config/binary fingerprint defines the cache identity. Generated metadata includes room grammar, lighting, object counts, and camera provenance; no external furniture assets or humans are used in this cohort.",y,size=9,width=104)
    report.save(fig)

    fig=report.new("Compute and training efficiency","Measured wall time and memory; device activity includes desktop work")
    paragraph(fig,summary["machine"]["commands"]["gpu"]+"; Burn 0.21.0 / CUDA fusion / F32.",.85,size=8,width=125)
    ts=np.array([r["elapsed_seconds"] for r in telemetry]);stride=max(1,len(ts)//1800)
    ax=fig.add_axes([.12,.63,.80,.20]);ax.plot(ts[::stride]/60,[r.get("device_gpu_percent",np.nan) for r in telemetry[::stride]],color=TEAL,linewidth=.8)
    ax.axvline(train["prepare_seconds"]/60,color=GRAY,ls="--",label="preparation ends");ax.set(xlabel="Run minutes",ylabel="Device GPU activity (%)",ylim=(0,105));ax.legend(fontsize=8)
    ax=fig.add_axes([.12,.36,.80,.20]);ax.plot(ts[::stride]/60,[r.get("process_vram_mib",np.nan)/1024 for r in telemetry[::stride]],color=BLUE,linewidth=1)
    ax.set(xlabel="Run minutes",ylabel="Process VRAM (GiB)")
    y=.29
    for p in [f"Warm median / p90 step: {perf['warm_step_median_ms']:.2f} / {perf['warm_step_p90_ms']:.2f} ms. Warm throughput: {perf['warm_examples_per_second']:.1f} target examples/s. End-to-end throughput including preparation/probes/checkpoints: {perf['end_to_end_examples_per_second']:.1f} examples/s.",f"Preparation: {train['prepare_seconds']:.1f} s. Training-process total: {train['run_seconds']:.1f} s. Peak process VRAM: {perf['peak_process_vram_mib']/1024:.2f} GiB. Steady device activity: {perf['steady_device_gpu_percent_mean']:.1f}% over {perf['steady_samples']} samples; mean power {perf['steady_device_power_w_mean']:.1f} W.","Telemetry is sampled at 1 Hz. Warm report statistics exclude the first ten updates; compilation outliers and clock/desktop effects can remain. GPU activity is neither achieved FLOP utilization nor measured occupancy. No kernel-level profiler was used."]:
        y=paragraph(fig,p,y,size=9,width=108)
    report.save(fig)

    fig=report.new("Step timing and schedule","Throughput includes the masked encoder, decoder, backward pass, clipping, and AdamW")
    x=np.array([s["step"] for s in steps]); elapsed=np.array([s["elapsed_ms_precise"] for s in steps])
    window=min(500,len(steps))
    ax=fig.add_axes([.12,.62,.80,.21]);ax.plot(x[window-1:],cfg["batch_size"]*1000/moving(elapsed,window),color=TEAL)
    if summary["resource_events"]:
        ax.axvline(45000,color=ORANGE,ls="--",lw=1,label="Concurrent rustc activity observed")
        ax.legend(fontsize=7,loc="lower left")
    ax.set(xlabel="Optimizer step",ylabel="Target examples / second",title=f"{window}-step timing windows (excludes probes/checkpoints)")
    ax=fig.add_axes([.12,.35,.80,.20]);ax.plot(x,[s["learning_rate"] for s in steps],color=BLUE)
    ax.set(xlabel="Optimizer step",ylabel="Learning rate");ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    y=.28
    for p in [f"The learning-rate schedule was fixed at {cfg['schedule']['decay_steps']:,} steps before training. Warmup lasts {cfg['schedule']['warmup_steps']:,} steps, followed by cosine decay to {cfg['schedule']['min_lr_ratio']*100:.1f}% of the peak. No schedule adjustment follows evaluation results.",f"Cached full-view feature tensors account for {train['cached_full_feature_bytes']/1024**3:.2f} GiB; resident RGB plus normalized copies account for {train['resident_rgb_bytes']/1024**3:.2f} GiB. These tensor-byte counters exclude model/optimizer state, activation/work buffers, and allocator reservations. Telemetry supplies the measured process peak.","The diagnostic phase-timing mode is disabled for the main run to avoid extra synchronization between phases. Per-step totals are synchronized by recorded scalar metrics and gradients. Cold start, cache preparation, fixed probes, and checkpoint serialization are reflected in end-to-end throughput."]:
        y=paragraph(fig,p,y,size=9,width=106)
    report.save(fig)

    fig=report.new("Convergence and optimizer behavior","All curves come from saved metrics; fixed probes do not change sampled inputs")
    x=np.array([s["step"] for s in steps]);window=max(1,len(x)//250)
    ax=fig.add_axes([.12,.61,.80,.22])
    for key,color in [("cross",BLUE),("mae",ORANGE),("ri",TEAL)]:
        ax.plot(x[window-1:],moving([s[key] for s in steps],window),label=key,color=color)
    ax.set(xlabel="Optimizer step",ylabel="Loss",title=f"Training components ({window}-step moving mean)");ax.legend()
    ax=fig.add_axes([.12,.31,.80,.22])
    for split,color in [("train",BLUE),("validation",ORANGE)]:
        ax.plot([p["step"] for p in train["probes"]],[p[split]["total"] for p in train["probes"]],"o-",ms=3,label=split,color=color)
    ax.set(xlabel="Optimizer step",ylabel="Total loss",title=f"Fixed panels, {train['probes'][0]['examples_per_split']} examples per split");ax.legend()
    zoom=ax.inset_axes([.40,.38,.56,.56])
    later=[p for p in train["probes"] if p["step"]>=2000]
    for split,color in [("train",BLUE),("validation",ORANGE)]:
        zoom.plot([p["step"] for p in later],[p[split]["total"] for p in later],color=color,lw=1)
    zoom.set_title("After step 2,000",fontsize=7);zoom.tick_params(labelsize=6)
    y=paragraph(fig,f"Gradient clipping occurred on {perf['clipped_fraction']*100:.1f}% of updates. The dataset received {visits:,} target-example visits ({epochs:.1f} room/view passes). Masks vary by step and are shared within each batch.",.24,size=9,width=106)
    late=perf["late_training_windows"]
    y=paragraph(fig,f"Last two {perf['late_window_steps']:,}-update windows: total loss {late['total']['previous']:.4f} → {late['total']['terminal']:.4f}; cross MSE {late['cross']['previous']:.4f} → {late['cross']['terminal']:.4f}. These are descriptive training averages under a changing learning rate, not independent confidence intervals or proof of full convergence.",y,size=9,width=106)
    paragraph(fig,"Held-out reconstruction, reference utility, and geometric ranking are evaluated separately below.",y,size=9,width=106)
    report.save(fig)

    fig=report.new("Held-out evaluation","One fixed evaluation mask; all target views; unknown geometry excluded")
    rows=[]
    for name in ["validation_initial","validation_final","test_initial","test_final"]:
        e=ev[name]; m=e["covisibility"]
        rows.append([name.replace("validation","val").replace("_"," / "),f"{e['mean_cross_mse']:.4f}",f"{e['mean_mae_mse']:.4f}",f"{e['mean_ri_loss']:.4f}",f"{m['average_precision']:.4f}",f"{m['auroc']:.4f}"])
    table(fig,rows,["Split / checkpoint","Cross MSE","MAE MSE","RI loss","AP","AUROC"],[.05,.64,.90,.20],[.25,.15,.15,.15,.15,.15],8)
    y=.60
    y=paragraph(fig,f"Final validation: {val['evaluated_target_views']} target views, {val['covisibility']['pixels']:,} ranked pixels, {val['unknown_pixels']:,} unknown pixels. Final test: {test['evaluated_target_views']} target views, {test['covisibility']['pixels']:,} ranked pixels, {test['unknown_pixels']:,} unknown pixels.",y)
    y=paragraph(fig,f"Visible-pixel prevalence / random-ranking AP baseline: validation {val['ap_prevalence_baseline']:.4f}; test {test['ap_prevalence_baseline']:.4f}. AUROC chance is 0.5. Visibility means visible in at least one of the two correct references. AP ties are grouped before threshold integration.",y)
    y=paragraph(fig,"Room-level bootstrap intervals (2,000 resamples, fixed seed) account for shared-room dependence. The macro endpoint averages valid target-view AUROCs within each room, then averages rooms; it differs from pooled pixel AUROC. Undefined single-class target AUROCs are omitted.",y)
    rows=[]
    for split,e in [("Validation",val),("Test",test)]:
        rows.append([split,fmt_ci(e["room_metrics"]["auroc"]["bootstrap"]),fmt_ci(e["room_metrics"]["cross_mse"]["bootstrap"])])
    table(fig,rows,["Split","Macro AUROC [95% room CI]","Cross MSE [95% room CI]"],[.05,.20,.9,.15],[.2,.4,.4],8)
    paragraph(fig,"These intervals describe variation among synthetic rooms in this fixed cohort. They do not include training-seed variation, encoder conversion uncertainty, or real-world domain shift.",.14,size=9,width=106)
    report.save(fig)

    fig=report.new("Reconstruction sanity check","Post-hoc zero-prediction baseline; no effect on training or endpoint selection")
    baseline=summary["zero_baseline"]
    ax=fig.add_axes([.14,.54,.77,.29]);names=["Constant zero","Initial cross","Final cross","Final MAE"]
    vals=[baseline["test"]["zero_normalized_mse"],ev["test_initial"]["mean_cross_mse"],test["mean_cross_mse"],test["mean_mae_mse"]]
    ax.bar(names,vals,color=[GRAY,"#93c5fd",TEAL,ORANGE]);ax.set(ylabel="Test masked normalized-pixel MSE")
    for i,v in enumerate(vals):ax.text(i,v,f" {v:.4f}",ha="center",va="bottom")
    y=.47
    for p in [f"A constant-zero prediction in the normalized RGB coordinate system has MSE {baseline['validation']['zero_normalized_mse']:.4f} on validation and {baseline['test']['zero_normalized_mse']:.4f} on test. This CPU calculation uses the same saved evaluation mask and all held-out target views. Lower model error than this baseline shows reconstruction learning beyond shrinking an initially random output toward zero.",f"The fraction of patches with low spatial contrast is {baseline['validation']['low_spatial_contrast_fraction']*100:.1f}% on validation and {baseline['test']['low_spatial_contrast_fraction']*100:.1f}% on test. Low contrast means within-patch spatial standard deviation, pooled across channels, below 0.01 sRGB units. This is a descriptive cohort statistic, not a co-visibility label or a causal explanation of the result.","These checks were added during the run to interpret the observed reconstruction losses. They are exploratory diagnostics and were not used to select data, hyperparameters, or a checkpoint. Reference utility is assessed separately with the matched unrelated-room intervention."]:
        y=paragraph(fig,p,y,size=10,width=96)
    report.save(fig)

    fig=report.new("Does the model use its references?","Paired endpoint control; no retraining and no test-driven model selection")
    y=.84
    for p in ["The intervention replaces both reference images with views from the next held-out room, wrapping at the end. Target pixels, target mask, model weights, and evaluated loss pixels remain fixed. Geometry labels are held to the original correct-reference scene for diagnostic score comparisons; the primary control endpoint is reconstruction error.",f"Correct-reference cross MSE: {test['mean_cross_mse']:.5f}. Unrelated-reference cross MSE: {ev['test_unrelated']['mean_cross_mse']:.5f}. Target-only MAE MSE: {test['mean_mae_mse']:.5f}. With original geometry labels held fixed, RI AUROC is {test['covisibility']['auroc']:.4f} with correct references and {ev['test_unrelated']['covisibility']['auroc']:.4f} with unrelated references (diagnostic only).",f"Paired room mean (unrelated minus correct cross MSE): {fmt_ci(controls['unrelated_minus_correct_mse'],6)}. Positive values favor correct references. Paired room mean (MAE minus correct cross MSE): {fmt_ci(controls['mae_minus_cross_mse'],6)}. Positive values favor cross-view over monocular reconstruction.",f"The MAE branch's largest per-target change under the reference intervention is {controls['max_mae_control_difference']:.8f}. This is an isolation check: target-only reconstruction should not depend on supplied references."]:
        y=paragraph(fig,p,y)
    ax=fig.add_axes([.15,.22,.72,.23]);names=["Correct references","Unrelated room","Target-only MAE"]
    values=[test["mean_cross_mse"],ev["test_unrelated"]["mean_cross_mse"],test["mean_mae_mse"]]
    ax.bar(names,values,color=[TEAL,ORANGE,BLUE]);ax.set(ylabel="Masked normalized-pixel MSE")
    for i,v in enumerate(values):ax.text(i,v,f" {v:.4f}",ha="center",va="bottom")
    paragraph(fig,"Reference sensitivity is necessary but insufficient for correspondence learning. A useful result also needs consistent reconstruction advantage, informative RI ranking, and confirmation across seeds/data domains. The diagnostic extension attends jointly to two references; it is not a pairwise Gekko checkpoint reproduction.",.15,size=9,width=106)
    report.save(fig)

    for number,directory in enumerate(test["samples"],1):
        a=sample_arrays(directory);meta=a["meta"];m=meta["metrics"]
        room=next(r for r in inv["rooms"] if r["seed"]==meta["room_seed"])
        fig=report.new(f"Annotated held-out sample {number}",f"{room['layout']} • seed {meta['room_seed']} • {room['objects']} objects • target {meta['target_view']} • references {meta['reference_views']} • predefined selection",landscape=True)
        grid=fig.add_gridspec(3,4,left=.055,right=.94,bottom=.13,top=.84,wspace=.12,hspace=.23)
        images=[(a["target"],"Target RGB",None,None),(a["masked"],"Reconstruction input (75% hidden)",None,None),
                (a["references"][0],"Reference 1",None,None),(a["references"][1],"Reference 2",None,None),
                (a["mae"],"MAE reconstruction *",None,None),(a["cross"],"Cross-view reconstruction *",None,None),
                (np.where(a["hidden"],a["mae_error"],np.nan),"MAE error (hidden pixels)", "magma",(0,2)),(np.where(a["hidden"],a["cross_error"],np.nan),"Cross error (hidden pixels)","magma",(0,2)),
                (np.where(a["visibility"]==255,2,a["visibility"]),"Geometric visibility",ListedColormap(["#dc2626","#16a34a","#94a3b8"]),(0,2)),
                (a["ri"],"Predicted RI (unbounded score)","RdBu",(-.5,.5)),(np.where(a["hidden"],a["empirical_ri"],np.nan),"Observed RI (hidden pixels)","RdBu",(-.5,.5)),
                (np.where(a["hidden"],a["mae_error"]-a["cross_error"],np.nan),"Masked MAE − cross error","RdBu",(-.5,.5))]
        for i,(data,title,cmap,limits) in enumerate(images):
            ax=fig.add_subplot(grid[i//4,i%4]);kw={} if limits is None else dict(vmin=limits[0],vmax=limits[1])
            im=ax.imshow(data,cmap=cmap,interpolation="nearest",**kw);ax.set_title(title,fontsize=9);ax.axis("off")
            if i in [6,7,9,10,11]:fig.colorbar(im,ax=ax,fraction=.046,pad=.015).ax.tick_params(labelsize=7)
        auc=m["covisibility"]["auroc"]
        auc_text="undefined" if auc is None else f"{auc:.3f}"
        fig.text(.055,.085,f"Cross MSE {m['cross_mse']:.4f}  |  MAE MSE {m['mae_mse']:.4f}  |  pixel AUROC {auc_text}.  Visibility: green visible, red not visible, gray unknown.",fontsize=8)
        fig.text(.055,.059,"* Normalized reconstructions use ground-truth target patch mean/std for display; this is an oracle visualization. Error/RI color scales clip at shown bounds; metrics use unclipped values.",fontsize=7)
        report.save(fig)

    fig=report.new("Limitations and next decisions","Interpretation boundaries matter more than a headline loss")
    y=.84
    items=[
        "One training seed, one synthetic rendering cohort, one evaluation mask. Independent repeated seeds and real indoor imagery are required before making a generalization claim.",
        "Encoder package hashes and required tensors are checked. Official native-image parity, original source-checkpoint provenance, and conversion error remain open. The imported encoder core is unchanged; the decoder is trained from scratch.",
        "Geometry labels use a finite nearest-depth tolerance. Thin surfaces, transparency, antialiasing, and boundaries can be ambiguous. Unknowns are excluded rather than treated as negatives.",
        "Normalized RGB loss removes per-patch mean/scale information. The annotated reconstruction pages restore that information using target statistics for visualization, so they must not be interpreted as deployable RGB reconstruction quality or PSNR.",
        "The reference intervention measures sensitivity at the terminal checkpoint. It does not establish explicit matches, calibrated co-visibility probabilities, pose recovery, or 3D reconstruction accuracy.",
        "Resolution screens are short; actual-run timing is reported separately. Concurrent Rust compilation was observed during a slowdown near step 45,000, with no priority or GPU-setting changes. This is a shared-workstation measurement, not an isolated hardware benchmark. Activity and sampled VRAM do not establish occupancy or achieved FLOP utilization.",
        "The four predefined examples show blocky reconstructions and limited fine structure, even with target statistics restored for display. Two selected target views have AUROC below 0.5 (0.487 and 0.458). Their RI maps do not cleanly trace thin occlusion boundaries. These examples accompany, rather than replace, the complete 192-view test statistics.",
        "Next decisions: qualify encoder image parity and repeat independent training seeds. Add matched-compute MAE-only and CroCo-only trainings, plus one-reference controls, before a paper-level comparison. Then evaluate real indoor transfer and sparse reference tokens. More hardware is not justified by this pilot alone.",
    ]
    for i,p in enumerate(items,1):y=paragraph(fig,f"{i}. {p}",y,size=9,width=106)
    report.save(fig)

    fig=report.new("Reproduction and artifact index","All datasets, weights, checkpoints, logs, annotations, and this PDF are under .data")
    y=.84
    for p in [f"Dataset ID: {inv['dataset_id']}",f"Run directory: {summary['run']}",f"Completed endpoint: checkpoint-{train['completed_steps']:06}. Decoder and AdamW states have checksums and a config/source/backend identity; periodic checkpoints and step-zero weights are preserved.","Protocol and resolved TOML configs: .data/pilot-02/protocol.toml, .data/pilot-02/sizing.json, configs/archive/pilot-02/pilot02-main.toml, configs/data/capture-pilot02-main.toml. Each execution stage has a command plan, ledger.json, combined output logs, and GPU telemetry.","Numerical source tables: .data/pilot-02/report-summary.json, geometry-audit.json, dataset-inventory.json, and per-target evaluation JSON in the run directory. Sample exports include HWC RGB, patch-major predictions, visibility labels, and explicit array-layout metadata.","Verification: .data/pilot-02/verification.json records runtime source/binary checks, terminal checkpoint checksums, split/view coverage, and four independent exported-array loss calculations (maximum difference below 6e-8).", "Preserved runtime: .data/pilot-02/bin/gekko. Source snapshot: .data/pilot-02/source-snapshot.tar.gz. Python package versions: .data/pilot-02/analysis-environment.txt. Neither report regeneration nor verification launches GPU work.","Regenerate: .data/analysis-venv/bin/python tools/legacy/render_pilot_report.py --study .data/pilot-02 --run .data/runs/pilot-02-main --dataset <dataset path>. This reads saved artifacts; it does not train or select a model.","Local source: src/pilot.rs, src/batch.rs, src/model.rs, src/loss.rs, src/eval.rs, crates/gekko_data, tools/zeroverse_capture. The V-JEPA import manifest records revision 939abcea4648fd2ad0e12cb6d7bf4874f6bdf871. Package pins and archive checksums are in the independent capture Cargo.lock.","Context and methods: Gekko paper https://arxiv.org/abs/2609.01530; original planning/source audit in docs/source-audit.md and docs/architecture.md. This implementation is an experimental set-conditioned extension and uses the released-code clamped RI objective."]:
        y=paragraph(fig,p,y,size=9,width=106)
    report.save(fig)
    report.close()
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True);parser.add_argument("--run",type=Path,required=True);parser.add_argument("--dataset",type=Path,required=True)
    args=parser.parse_args()
    assert args.study.resolve().is_relative_to(Path(".data").resolve())
    summary,telemetry=collect(args.study,args.run,args.dataset)
    print(build_pdf(args.study,summary,telemetry))


if __name__=="__main__":main()

# Pilot 19: geometry and completion balance

Registered after the completed Pilot 18 decision, before any Pilot 19 training.
Pilot 18 improved synthetic viewpoint error and camera recovery but lost the
matched control's small neighbor-detail gain. Its adjacent-correlation drop of
0.005048509 exceeds the fixed 0.005 tolerance. The parent remains selected.

Hypothesis: halving actual-view geometry NLL strength permits more completion
detail learning while retaining useful matching and camera improvements. This is
one new hyperparameter experiment on reused development data, not confirmation
on an untouched test set. The failed 0.1-weight result remains recorded.

## Fixed run and reused controls

Run one **4,096-update** candidate with **geometry weight 0.05**, initialized from
the same Pilot 13 parent, with both optimizers reset. Use the immutable Pilot 18
CUDA trainer (`b411843bf402e02fc04bbf68d4a2b04dc8e0556f00234cb55990f18fb534843e`,
source `d61b84be4a50e6629edc55e24d01241181db09f4`). Keep the full 8,192-room
cohort, batch 16, seed 853, 128-step warmup, 4,096-step cosine horizon, learning
rates, all other losses, label cache, masks, 151-tensor encoder stage and final
endpoint selection unchanged. The trainer cap remains 10,000 seconds and the
command cap 10,100 seconds. Never choose an intermediate checkpoint.

Reuse the completed Pilot 18 control
`02db25fcd1f2cb808619558ea8b9e7123d1d93bcf5b62187bc693271366299e4`
and original parent
`0639afa2e19d0e7ac5c9c1cc8caadc75d0dcf10f105ee2870607856b7a1ed5a7`.
They already execute the exact comparison recipe, source and initial weights.
Native selection must verify matching full sample streams, schedules and source
identities again. Reuse their complete common-mask completion exports and
geometric/pose reports with pinned hashes. Do not rerun a control merely to
obtain a more favorable realization. The rejected 0.1 candidate cannot become
the parent or replace either control.

## Budget and order

Charge this study to **`.data/pilot-18/budget.json`**, within the same user-approved
43,200 GPU-command seconds. No new allowance is created and the old 96.281 seconds
remain separate. First finish the registered warm-dispatch diagnostic and the
Pilot 18 report. Require no other experiment GPU command to be running.

The graph and batch shape are unchanged from Pilot 18 geometry, which completed
in 4,890.28 seconds with warm p95 1.25305 seconds. Reuse its already registered
native forecast of **7,274.14 seconds**, based on the higher preflight p95 of
1.26746 seconds, 25% timing margin, measured preparation/finalization and a
600-second periodic-I/O margin. Pin that report before launch. Require it to fit the
10,000-second trainer cap and the remaining ledger with a **7,200-second evaluation
reserve**. No additional short throughput preflight is needed for a scalar loss
weight change with this completed same-shape run. Preserve failures and do not
silently reduce the horizon or retry the candidate.

## Unchanged retention criteria

Use exactly Pilot 18's eight conjunctive quality checks: hidden MSE within 1% of
the better parent/control, references beneficial, view AEPE at least 5% below
both, nondecreasing PCK8 against both, centered MSE within 1% of control, centered
spatial and adjacent-difference correlations each within 0.005 of control, and
mean pose AUC@10 at least both parent and control. The **0.005 detail tolerance is
unchanged**. Any failure retains the published parent.

Assess the same first 64 validation rooms, 90% random masking, seed 853, two
references and exact fixed-teacher target arrays. Matching uses the same first
32 validation rooms. Calibrated eight-point pose retains seeds 871–878, all three
readouts, all failures, the same pixel threshold and RANSAC settings. Native
room-bootstrap intervals and seed ranges remain mandatory. The generic native
selector binds the expected positive geometry weight to the pinned registered
recipe and actual execution; allowing 0.05 does not alter any quality threshold.

This is an adaptive follow-up motivated by reused synthetic validation. Real
benchmarks cannot choose the candidate; independent qualification and an expanded
RGB/camera-head study require a synthetic pass. No noncommercial weights enter
training. A page/paper must describe one final foundation run and its own audited
heads. Do not automatically consume unused budget.

Before launch, native selection replay reproduced the complete Pilot 18 decision
exactly. A deliberately mismatched expected geometry weight was rejected. Reused
assessment bundles may differ only in their model roster; all scientific settings,
teacher arrays, masks, room/view identities and individual checkpoint bindings
remain checked. Fifty evaluator tests and strict workspace Clippy passed.

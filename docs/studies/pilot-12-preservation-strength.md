# Conditional follow-up: stronger final-feature preservation

Registered while the primary weight-4 candidate is running, before inspecting
its first scheduled quality validation. The completed controls show a clear
tradeoff: unrestricted full adaptation reduces local transform error from
7.4364 to 3.5863 pixels while worsening completion MSE from 0.1754603 to
0.1813925. This motivates a conditional coefficient check, not a new architecture.

## Trigger and budget

Run only if the primary candidate fails the **1% completion-retention gate**
but passes every other registered gate: lower completion MSE than unrestricted
full, positive reference benefit, at least 10% lower transform error than tail,
and no PCK8 loss. Require its exact common-parent parameter check, unchanged
teacher/anchor probes, full gradient coverage and amended numerical bounds.
If it qualifies, follow the original fixed 4,096-update continuation instead;
do not run this follow-up. If another primary gate fails, skip this follow-up.

Use only the residual of Pilot 12's 16,723.149945212062-command-second ceiling,
which itself is the residual of the existing 12-hour authorization. Complete
the original three-arm exports and single-run report first. Then require a
forecast of 2,048 times primary-candidate p95 update time times 1.10 plus
600 seconds, and a further **1,500-second evaluation reserve**, to fit.
No new GPU allowance, overlapping jobs, shorter horizon or automatic allocation
retry is authorized by this protocol. Failed preflights remain charged.

## Candidate and unchanged controls

Use the same sealed instrumented trainer and original Pilot 11 main parent,
fixed anchor, 8,192 rooms, seed 823, 2,048 updates, 100-update warmup, batch 16,
trunk LR 4e-5 and encoder LR 8e-7. The **only recipe change is preservation
weight 16.0 instead of 4.0**. Both optimizers reset. Keep the 4,080-second
internal limit and 4,200-second command cap, the same ordered exposures,
32-room validation, masks, teacher, spatial/latent/RI/warp objectives and
32-room transform diagnostic with seed 1000823.

Reuse the already completed tail and unrestricted-full controls. Their
training source, parent, sample order, horizon and all recipe fields must
match through the same native selector. This is a conditional development
follow-up using common controls, not an independent replication or a new
three-seed study. The selector's expected coefficient is explicitly 16.0;
its default stays 4.0 for the original study. Never infer the expected value
from the candidate's observed recipe.

All quality gates and numerical bounds are unchanged. No coefficient beyond
16, threshold relaxation or intermediate checkpoint selection is planned.
There is **no further training continuation** for this follow-up. Fix its
2,048-update final endpoint regardless of whether it qualifies.

## Evaluation and reporting

Seal the synthetic decision before any external export. Score every one of
the 580 HPatches, 3,365 ETH3D and 186 TUM development pairs for this endpoint,
including when it fails. Use the same operators, populations and confidence
intervals as the original study. External scores do not choose the coefficient
or endpoint. Prior reused cohorts stay development data.

After fixing the endpoint, capture another fresh 128-room, four-view cohort
with seed **2610091000**, disjoint from all existing manifests, and evaluate
all 512 targets with mask seed **831**. A native single-run page/PDF reports
this candidate's metrics, annotated samples, limitations and numerical floor.
It contains no private-arm comparisons; those remain in this internal study.
Label a rejected candidate diagnostic. Preserve the original weight-4 page,
PDF, selection, sources and all outputs unchanged. No SOTA or RGB blur claim
follows from synthetic retention alone.

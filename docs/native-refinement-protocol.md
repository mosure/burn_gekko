# Native spatial-refinement experiment

Registered before training. Budget: remaining Pilot 07 allowance, with the same
shared `.data/pilot-07/budget.json`; no new 12-hour allowance. All GPU commands,
including compatibility checks, count. CPU reorganization and reporting do not.

One fixed 3,000-update weights-only phase begins from the audited selected
checkpoint `61ecd81f7d9c4094ff1fa120e761176301db14e4a22e056cc966bf8616a81bf6`.
Both optimizers restart explicitly because the source identity changed. The
fixed MIT V-JEPA teacher and self-supervised latent/affinity objectives remain.
The appearance encoder from public Gekko is excluded. New block-6 affinity
targets are anchored to this phase's fixed warm-start encoder.

The hypothesis is that smaller fusion updates and limiting encoder adaptation
to the last two blocks preserve the strong intermediate spatial representation
while refining the decoder. Learning rate is 5e-5, encoder ratio 0.01, with the
existing validation gate. Training uses 8,192 cached rooms, 256x256 images,
two references and 90% target masking. This is one controlled continuation,
not a claim that the fusion weakness is fixed.

The selected checkpoint is the fixed final endpoint, irrespective of observed
quality. Existing validation rooms and previously inspected ETH3D/HPatches are
development evidence. A new disjoint procedural cohort will be evaluated after
the endpoint is frozen, if the remaining command budget permits. The paper/page
will present the selected experiment alone, with within-model information-set
controls. Failed gates and untrained heads remain visible.

Before training, the reorganized native evaluator reproduced eight target views
and all 80 float/label sample arrays bit-for-bit from the sealed checkpoint;
the exact array count is recorded in the machine-readable parity receipt.
Source reorganization does not authorize exact optimizer resume with a changed
source identity. Sealed old binaries and source archives remain available.

# Pilot 19: conditional real-view qualification

Registered before the Pilot 19 synthetic selection. Execute only after all
synthetic retention checks pass, without revising the candidate or selection
criteria using external outcomes. These are repeatedly inspected development
benchmarks, not an untouched test set or evidence of SotA.

Reuse the complete established RGB-only export protocols: all **580 HPatches
pairs** (295 viewpoint pairs primary), all **3,365 canonical ETH3D pairs**, and all
**186 TUM pairs** at 256 pixels. Pin the same input manifests, preprocessing,
block-6 spatial readouts, canonical score arrays, hard/local/mutual definitions
and inference binaries from Pilot 13. Replace only the selected checkpoint and
the output/selection-record paths. The decoder architecture is unchanged.
TOML recipes and final weight hashes are sealed before any GPU export.

HPatches, ETH3D and TUM have respective GPU command caps of **240, 700 and 180
seconds**. Keep ETH3D's existing 570-second internal cap; an incomplete export
fails rather than becoming a smaller reported cohort. All commands use the same
`.data/pilot-18/budget.json`, with at least all three command caps plus a
3,600-second reserve required before launch. No concurrent experiment GPU job,
threshold sweep, repeated export or hidden failure is allowed.

Retain all five existing correspondence contrasts for each matching benchmark.
For TUM, retain the original eight-point score, then run both eight- and
five-point localization panels across seeds 781–788, all three descriptor routes
and both recorded subpixel/hard coordinate readouts. Use the same 2,048 maximum /
64 minimum RANSAC trials, confidence 0.999, 12 inliers, 3-original-pixel Sampson
threshold, 1 cm eligibility and signed translation convention. Geometry and
intrinsics enter only subsequent CPU scoring/pose fitting, never RGB inference.
Failures remain in denominators; do not select the best solver seed.

Within this candidate, a useful transfer result requires all existing HPatches
and ETH3D contrasts to pass, and five-point recorded-coordinate pair conditioning
to pass the existing all-sequence camera gates against **both** same-image and
encoder controls for at least **six of eight** solver seeds. The point and seed
range of every readout remain visible regardless of these checks. These are
engineering qualification criteria, not a new public benchmark standard.

Report this checkpoint alone with its own information controls. The foundation
retention decision, learned RGB/calibration heads, and calibrated solver probes
remain distinct. Failed transfer prevents a claim that the original fusion
weakness is resolved; better synthetic loss or PSNR cannot waive it.

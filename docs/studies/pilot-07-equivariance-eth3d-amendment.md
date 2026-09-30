# ETH3D execution amendment for the equivariance preflight

Registered 2026-09-30 after HPatches scoring and before ETH3D inference for this
checkpoint. No model weights, primary readout, control readout or matching
operator changes. This is an execution optimization, not an additional fit.

Checkpoint: `996fedf858d6c5e1fe0cdaabd8539e87ff3bd7e6041625ea2981b41056be5e73`.
The existing ledger has 243.424 command seconds remaining. The broad ETH3D
exporter took about 368 seconds in the preceding study; that would not fit.

The focused exporter computes only the already registered three readouts:
`spatial_residual_conditional`, `spatial_self_conditional`, and
`student_l06_centered_conditional`. It reuses block-6 features already present
in the model's encoder output and avoids fixed-teacher, repeated encoder and
unused attention/readout work. Model inputs and point labels are unchanged;
geometric labels remain unavailable to GPU inference.

Before emitting a complete result, the first eight pairs must match the original
export implementation exactly in both predicted indices and mutual-match flags.
After 64 pairs, record the warm p95 pair latency. Stop if projected total time
(remaining pairs at p95, plus 15% margin and two seconds for output) exceeds
210 seconds. The command has a separate 220-second watchdog. Any abort retains
partial predictions and consumes the ledger; partial exports are not scored or
represented as the complete 3,365-pair protocol.

The selected checkpoint, all three readouts, output flag and source/binary hashes
are sealed before inference. Previously observed ETH3D data remain development
data. Runtime qualification and benchmark quality are separate results.

# burn_gekko_inference

The native and browser demo use the same RGB-only model, preprocessing, camera
heads and portable Rust scores. Model bytes are verified before deserialization.
Tensor readbacks are asynchronous; no native training/data pipeline is pulled
into the browser. The full target is used only by the separately labeled dense
camera and relative-improvement routes, and by RGB scoring after completion.

`gekko-demo-export` packages the selected foundation and attached heads into
checksummed chunks with their original float32 precision and license notices.

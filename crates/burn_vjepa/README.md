# burn_vjepa

V-JEPA 2.1 image/video encoders, sparse token masks and audited checkpoint loading
for Burn 0.21. This is the small encoder fork maintained in
[burn_gekko](https://github.com/mosure/burn_gekko), derived from
[mosure/burn_jepa](https://github.com/mosure/burn_jepa). It is separate from the
unrelated `burn_jepa` package on crates.io.

```toml
[dependencies]
burn_vjepa = "0.1.0"
```

The default backend is NdArray; `cuda` and `wgpu` enable GPU backends. Optional
`sparse-patchify-cuda` / `sparse-patchify-wgpu` enable specialized patch projection.
Views are encoded independently using the native image path. Dense/sparse output,
checkpoint round trips and final-only gradient contracts have CPU regression tests.
No pretrained weights are included.

The six original file hashes and revision are in `UPSTREAM.json`; the local model
adaptation is recorded in `LOCAL_MODIFICATIONS.toml`. See `UPSTREAM.md` for source
attribution, numerical qualification and the separate pretrained-weight notice.
Code is MIT OR Apache-2.0; audited V-JEPA weight artifacts retain their own MIT notice.

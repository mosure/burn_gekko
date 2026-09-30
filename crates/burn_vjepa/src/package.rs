//! Local sharded Burnpack reader, adapted from upstream model_package.rs at the pinned revision.
use crate::{VJepa2_1Model, VJepaConfig};
use anyhow::{Result, ensure};
use burn::{
    module::{Module, ModuleMapper, Param},
    tensor::{Bytes, DType, FloatDType, Tensor, backend::Backend},
};
use burn_store::{BurnpackStore, ModuleAdapter, ModuleSnapshot, TensorSnapshot};
use std::{collections::BTreeSet, rc::Rc};

#[derive(Clone, Debug)]
struct F16LoadAdapter;
impl ModuleAdapter for F16LoadAdapter {
    fn adapt(&self, s: &TensorSnapshot) -> TensorSnapshot {
        if s.dtype != DType::F16 {
            return s.clone();
        }
        let original = s.clone_data_fn();
        TensorSnapshot::from_closure(
            Rc::new(move || Ok(original()?.convert_dtype(DType::F32))),
            DType::F32,
            s.shape.clone(),
            s.path_stack.clone().unwrap_or_default(),
            s.container_stack.clone().unwrap_or_default(),
            s.tensor_id.unwrap_or_default(),
        )
    }
    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}
struct F32Mapper;
impl<B: Backend> ModuleMapper<B> for F32Mapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        Param::from_mapped_value(id, tensor.cast(FloatDType::F32), mapper)
    }
}

/// Every parameter must occur exactly once across validated shards. Caller verifies file checksums.
pub fn load_burnpack_parts<B: Backend>(
    config: &VJepaConfig,
    parts: Vec<Vec<u8>>,
    device: &B::Device,
) -> Result<VJepa2_1Model<B>> {
    ensure!(!parts.is_empty(), "empty weights bundle");
    let mut model = VJepa2_1Model::new(config, device);
    let mut applied = BTreeSet::new();
    let mut missing = BTreeSet::new();
    for part in parts {
        let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(part)))
            .allow_partial(true)
            .validate(true)
            .with_from_adapter(F16LoadAdapter);
        let result = model.load_from(&mut store)?;
        ensure!(
            result.errors.is_empty() && result.skipped.is_empty(),
            "failed or skipped weight tensors: {:?}",
            result.errors
        );
        for name in result.applied {
            ensure!(
                applied.insert(name.clone()),
                "duplicate weight tensor {name}"
            );
        }
        missing.extend(result.missing.into_iter().map(|(name, _)| name));
    }
    missing.retain(|name| !applied.contains(name));
    ensure!(
        missing.is_empty(),
        "incomplete sharded checkpoint: {missing:?}"
    );
    Ok(model.map(&mut F32Mapper))
}

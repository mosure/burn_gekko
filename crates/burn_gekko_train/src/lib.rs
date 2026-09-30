//! Native training and prediction export. Historical CLI names remain adapters.
#[doc(hidden)]
pub mod provenance;
pub use burn_gekko::*;
pub mod data;
pub mod evaluation;
pub mod training;
pub use data::batch;
pub use data::encoding as encoder;
pub use evaluation::{
    assessment as latent_assess, correspondence, encoder as encoder_audit, eth3d,
    fusion as fusion_audit, hpatches, latent as latent_eval, rgb as eval,
};
pub use training::{
    hybrid as hybrid_pilot, latent as latent_pilot, pilot, preflight as train,
    reconstruction as e2e_pilot, transport as transport_pilot,
};

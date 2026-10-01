use crate::{Bundle, Inference, Request};
use burn::backend::wgpu::{
    CubeBackend, WgpuDevice, WgpuRuntime, graphics::WebGpu, init_setup_async,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WebInference {
    // Burn 0.21 fusion on browser WebGPU passed dense-camera parity but changed
    // completion PSNR by 1.77 dB. Direct CubeBackend matches native and NdArray;
    // keep the browser parity gate before enabling fusion here again.
    model: Inference<CubeBackend<WgpuRuntime, f32, i32, u32>>,
}

#[wasm_bindgen]
impl WebInference {
    /// Runs inside a dedicated Web Worker; the scene and editor stay interactive.
    pub async fn load(
        manifest: String,
        foundation: Vec<u8>,
        camera: Vec<u8>,
        rgb: Vec<u8>,
    ) -> Result<WebInference, JsValue> {
        console_error_panic_hook::set_once();
        let bundle = Bundle::parse(&manifest).map_err(error)?;
        let device = WgpuDevice::default();
        init_setup_async::<WebGpu>(&device, Default::default()).await;
        Ok(Self {
            model: Inference::load(bundle, foundation, camera, rgb, device).map_err(error)?,
        })
    }
    pub async fn infer(&self, request: JsValue) -> Result<JsValue, JsValue> {
        let request: Request = serde_wasm_bindgen::from_value(request).map_err(error)?;
        let result = self.model.run(request).await.map_err(error)?;
        serde_wasm_bindgen::to_value(&result).map_err(error)
    }
}
#[wasm_bindgen]
pub fn model_files(manifest: &str) -> Result<JsValue, JsValue> {
    let b = Bundle::parse(manifest).map_err(error)?;
    serde_wasm_bindgen::to_value(&b).map_err(error)
}
fn error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

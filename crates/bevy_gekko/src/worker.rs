use bevy::prelude::*;
use burn_gekko_inference::{InferenceOutput, Request, RgbInput};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Work {
    Load { root: String },
    Infer { request: Request },
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Progress { message: String },
    Ready,
    Output { result: Box<InferenceOutput> },
    Error { message: String },
    Uploaded { images: Vec<RgbInput> },
    UploadCancelled,
    UploadError { message: String },
}
#[derive(Resource)]
pub struct Worker {
    pub events: async_channel::Receiver<Event>,
    pub output: async_channel::Sender<Event>,
    #[cfg(not(target_arch = "wasm32"))]
    requests: async_channel::Sender<Work>,
}
impl Worker {
    pub fn new() -> Self {
        let (output, events) = async_channel::unbounded();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (requests, rx) = async_channel::bounded(1);
            let tx = output.clone();
            std::thread::spawn(move || {
                let errors = tx.clone();
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    pollster::block_on(native(rx, tx))
                }))
                .is_err()
                {
                    let _ = errors.try_send(Event::Error {
                        message: "Inference device failed. Restart the demo to retry.".into(),
                    });
                }
            });
            Self {
                events,
                output,
                requests,
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            web::start(output.clone());
            Self { events, output }
        }
    }
    pub fn send(&self, work: Work) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(e) = self.requests.try_send(work) {
            let _ = self.output.try_send(Event::Error {
                message: e.to_string(),
            });
        }
        #[cfg(target_arch = "wasm32")]
        if let Err(e) = web::send(&work) {
            let _ = self.output.try_send(Event::Error {
                message: format!("{e:?}"),
            });
        }
    }
    pub fn upload(&self) {
        let tx = self.output.clone();
        let task = async move {
            let files = rfd::AsyncFileDialog::new()
                .add_filter("Scene images", &["png", "jpg", "jpeg"])
                .pick_files()
                .await;
            let Some(files) = files else {
                let _ = tx.send(Event::UploadCancelled).await;
                return;
            };
            if !(2..=4).contains(&files.len()) {
                let _ = tx
                    .send(Event::UploadError {
                        message: "Choose two to four images of the same scene.".into(),
                    })
                    .await;
                return;
            }
            let mut images = Vec::new();
            for file in files {
                match RgbInput::decode(file.file_name(), &file.read().await, 256) {
                    Ok(v) => images.push(v),
                    Err(e) => {
                        let _ = tx
                            .send(Event::UploadError {
                                message: e.to_string(),
                            })
                            .await;
                        return;
                    }
                }
            }
            let _ = tx.send(Event::Uploaded { images }).await;
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || pollster::block_on(task));
        }
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(task);
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn native(rx: async_channel::Receiver<Work>, tx: async_channel::Sender<Event>) {
    use burn::backend::Wgpu;
    use burn_gekko_inference::{Bundle, Inference};
    let mut model: Option<Inference<Wgpu>> = None;
    while let Ok(work) = rx.recv().await {
        let result: anyhow::Result<Event> = match work {
            Work::Load { root } => (|| {
                let fetch = |name: &str| -> anyhow::Result<Vec<u8>> {
                    if root.starts_with("https://") || root.starts_with("http://") {
                        Ok(ureq::get(&format!("{}/{name}", root.trim_end_matches('/')))
                            .call()?
                            .body_mut()
                            .with_config()
                            .limit(64 * 1024 * 1024)
                            .read_to_vec()?)
                    } else {
                        Ok(std::fs::read(std::path::Path::new(&root).join(name))?)
                    }
                };
                let bundle = Bundle::parse(&String::from_utf8(fetch("manifest.toml")?)?)?;
                let mut foundation = Vec::new();
                for (i, file) in bundle.foundation.iter().enumerate() {
                    let _ = tx.try_send(Event::Progress {
                        message: format!("Loading weights {}/{}", i + 1, bundle.foundation.len()),
                    });
                    let bytes = fetch(&file.name)?;
                    file.verify(&bytes)?;
                    foundation.extend(bytes);
                }
                let camera = fetch(&bundle.camera.name)?;
                bundle.camera.verify(&camera)?;
                let rgb = fetch(&bundle.rgb.name)?;
                bundle.rgb.verify(&rgb)?;
                let _ = tx.try_send(Event::Progress {
                    message: "Initializing inference device…".into(),
                });
                model = Some(Inference::load(
                    bundle,
                    foundation,
                    camera,
                    rgb,
                    Default::default(),
                )?);
                Ok(Event::Ready)
            })(),
            Work::Infer { request } => {
                if let Some(model) = &model {
                    model.run(request).await.map(|result| Event::Output {
                        result: Box::new(result),
                    })
                } else {
                    Err(anyhow::anyhow!("Load the model first"))
                }
            }
        };
        let event = result.unwrap_or_else(|e| Event::Error {
            message: format!("{e:#}"),
        });
        if tx.send(event).await.is_err() {
            break;
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use wasm_bindgen::{JsCast, JsValue, closure::Closure};
    thread_local! { static WORKER: std::cell::RefCell<Option<web_sys::Worker>>=const {std::cell::RefCell::new(None)}; }
    pub fn start(tx: async_channel::Sender<Event>) {
        let options = web_sys::WorkerOptions::new();
        options.set_type(web_sys::WorkerType::Module);
        let worker = web_sys::Worker::new_with_options("inference-worker.js", &options);
        match worker {
            Ok(w) => {
                let errors = tx.clone();
                let receive =
                    Closure::<dyn FnMut(web_sys::MessageEvent)>::new(
                        move |event: web_sys::MessageEvent| {
                            let value = serde_wasm_bindgen::from_value(event.data())
                                .unwrap_or_else(|e| Event::Error {
                                    message: e.to_string(),
                                });
                            let _ = tx.try_send(value);
                        },
                    );
                w.set_onmessage(Some(receive.as_ref().unchecked_ref()));
                receive.forget();
                let error = Closure::<dyn FnMut(web_sys::ErrorEvent)>::new(
                    move |event: web_sys::ErrorEvent| {
                        let _ = errors.try_send(Event::Error {
                            message: event.message(),
                        });
                    },
                );
                w.set_onerror(Some(error.as_ref().unchecked_ref()));
                error.forget();
                WORKER.with(|s| *s.borrow_mut() = Some(w));
            }
            Err(e) => {
                let _ = tx.try_send(Event::Error {
                    message: format!("Worker unavailable: {e:?}"),
                });
            }
        }
    }
    pub fn send(work: &Work) -> Result<(), JsValue> {
        let value =
            serde_wasm_bindgen::to_value(work).map_err(|e| JsValue::from_str(&e.to_string()))?;
        WORKER.with(|s| {
            s.borrow()
                .as_ref()
                .ok_or_else(|| JsValue::from_str("Worker unavailable"))?
                .post_message(&value)
        })
    }
}

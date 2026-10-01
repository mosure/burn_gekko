mod app;
mod capture_image;
mod controls;
#[cfg(not(target_arch = "wasm32"))]
mod smoke;
mod state;
mod ui;
mod worker;
fn main() {
    app::run();
}

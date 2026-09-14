#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

pub mod hardware;
pub mod process;
pub mod setup;

fn main() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running clarity desktop application");
}

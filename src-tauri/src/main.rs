#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

pub mod boot;
pub mod commands;
pub mod migrate;
pub mod process;
pub mod setup;
pub mod state;

use tauri::Manager;

fn main() {
    let app_data_dir = state::AppState::default_app_data_dir();
    let resources_dir = state::AppState::resolve_resources_dir(None);

    // The logs directory has to exist before anything can fail, otherwise the
    // first error on a broken machine has nowhere to be written.
    let _ = std::fs::create_dir_all(app_data_dir.join("logs"));

    let app_state = state::AppState::new(app_data_dir, resources_dir);

    tauri::Builder::default()
        // Must be registered first: a second launch focuses the running window
        // instead of spawning a second backend on another port.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            // The exiting second launch is otherwise fully silent; leave a
            // trace so "won't open" reports can distinguish handoff from death.
            boot::append_boot_log(
                &state::AppState::default_app_data_dir(),
                "second launch handed off to the running instance",
            );
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            commands::get_app_config,
            commands::retry_boot,
        ])
        .setup(move |app| {
            let app_handle = app.handle().clone();

            // Ensure window icon is explicitly set to Clarity logo
            if let Some(window) = app_handle.get_webview_window("main") {
                if let Some(icon) = app_handle.default_window_icon() {
                    let _ = window.set_icon(icon.clone());
                }
            }

            // Provisioning and the backend can take minutes; never block setup.
            tauri::async_runtime::spawn(async move {
                boot::start(app_handle).await;
            });

            Ok(())
        })
        .on_window_event(|window, event| match event {
            tauri::WindowEvent::CloseRequested { .. } | tauri::WindowEvent::Destroyed => {
                let state = window.state::<state::AppState>();
                state.terminate_backend();
            }
            _ => {}
        })
        .run(tauri::generate_context!())
        .expect("error while running clarity desktop application");
}

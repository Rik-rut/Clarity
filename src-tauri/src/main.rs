#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

pub mod commands;
pub mod hardware;
pub mod process;
pub mod setup;
pub mod state;

use tauri::Manager;

fn main() {
    let app_data_dir = state::AppState::default_app_data_dir();
    let resources_dir = state::AppState::resolve_resources_dir(None);
    let app_state = state::AppState::new(app_data_dir.clone(), resources_dir);

    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            commands::get_setup_status,
            commands::start_setup,
            commands::retry_setup,
            commands::get_app_config,
            commands::launch_backend,
            commands::launch_main_app,
        ])
        .setup(move |app| {
            let app_handle = app.handle().clone();
            let is_complete = setup::is_setup_complete(&app_data_dir) || setup::has_dev_environment();

            tauri::async_runtime::spawn(async move {
                // Yield briefly to ensure webview window attachment on cold start
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;

                if is_complete {
                    if let Some(state) = app_handle.try_state::<state::AppState>() {
                        match commands::launch_backend_internal(&state).await {
                            Ok(port) => {
                                let url = format!("http://127.0.0.1:{}", port);
                                if let Err(e) = commands::navigate_window(&app_handle, &url) {
                                    eprintln!("Failed to navigate window to backend URL {}: {}", url, e);
                                }
                            }
                            Err(err) => {
                                eprintln!("Failed to launch backend on startup: {}", err);
                                if let Err(e) = commands::navigate_window(&app_handle, "setup.html") {
                                    eprintln!("Failed to navigate window to setup.html: {}", e);
                                }
                            }
                        }
                    }
                } else if let Err(e) = commands::navigate_window(&app_handle, "setup.html") {
                    eprintln!("Failed to navigate window to setup.html: {}", e);
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            match event {
                tauri::WindowEvent::CloseRequested { .. } | tauri::WindowEvent::Destroyed => {
                    let state = window.state::<state::AppState>();
                    state.terminate_backend();
                }
                _ => {}
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running clarity desktop application");
}

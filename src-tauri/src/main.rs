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
            let is_complete = setup::is_setup_complete(&app_data_dir);

            tauri::async_runtime::spawn(async move {
                if is_complete {
                    if let Some(state) = app_handle.try_state::<state::AppState>() {
                        match commands::launch_backend_internal(&state).await {
                            Ok(port) => {
                                let url = format!("http://127.0.0.1:{}", port);
                                let _ = commands::navigate_window(&app_handle, &url);
                            }
                            Err(err) => {
                                eprintln!("Failed to launch backend: {}", err);
                                let _ = commands::navigate_window(&app_handle, "setup.html");
                            }
                        }
                    }
                } else {
                    let _ = commands::navigate_window(&app_handle, "setup.html");
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

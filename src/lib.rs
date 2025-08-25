pub mod batch_processor;
pub mod config;
pub mod inference_client;
pub mod request_handler;
pub mod routes;
pub mod types;

use crate::config::AppConfig;
use crate::request_handler::RequestHandler;
use crate::types::ErrorResponse;
use rocket::config::LogLevel;
use rocket::serde::json::Json;
use rocket::{Build, Request, Rocket, catch, http::Status};
use std::sync::Arc;

/// Only catches errors that aren't explicitly handled,
/// has lower priority than custom responders, i.e., custom error handling bypasses this global catcher
#[catch(default)]
fn json_error_catcher(status: Status, _req: &Request) -> Json<ErrorResponse> {
    Json(ErrorResponse {
        error: format!("{}", status.reason().unwrap_or("Unknown Error")),
    })
}

/// Builds and configures a Rocket application instance.
/// Accessible from application as well as tests
pub async fn build_rocket(app_config: AppConfig) -> Rocket<Build> {
    let port = app_config.port;
    let log_level = if app_config.quiet_mode {
        LogLevel::Off // Silent Rocket (no startup messages)
    } else {
        LogLevel::Normal // Standard Rocket startup messages
    };

    // it's OK to fail earlier in this case, since it's App startup code
    let handler = Arc::new(
        RequestHandler::new(app_config)
            .await
            .expect("Failed to create RequestHandler"),
    );

    rocket::build()
        // once managed, this Arc<RequestHandler> instance is available to any route handler that declares it as a
        // parameter with the State guard
        // same Arc<RequestHandler> instance is shared across all requests
        .manage(handler)
        .mount("/", rocket::routes![routes::health, routes::embed])
        .register("/", rocket::catchers![json_error_catcher])
        .configure(rocket::Config {
            port,
            log_level,
            ..rocket::Config::default()
        })
}

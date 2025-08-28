use crate::request_handler::RequestHandler;
use crate::types::{EmbedRequest, EmbedResponse, ErrorResponse};
use rocket::http::Status;
use rocket::response::status::Custom;
use rocket::serde::json::Json;
use rocket::{State, get, post};
use std::sync::Arc;

/// POST /embed - Main embedding endpoint
///
/// Accepts a JSON request with string inputs and returns embeddings.
/// Requests are automatically batched for efficiency.
#[post("/embed", data = "<request>")]
pub async fn embed(
    request: Json<EmbedRequest>,
    request_handler: &State<Arc<RequestHandler>>,
) -> Result<Json<EmbedResponse>, Custom<Json<ErrorResponse>>> {
    if request.inputs.is_empty() {
        return Err(Custom(
            Status::BadRequest,
            Json(ErrorResponse {
                error: "`inputs` can't be empty".to_string(),
            }),
        ));
    }

    if request.inputs.len() > request_handler.config.max_inference_inputs {
        return Err(Custom(
            Status::PayloadTooLarge,
            Json(ErrorResponse {
                error: format!(
                    "`inputs` can't be greater than {}",
                    request_handler.config.max_inference_inputs
                ),
            }),
        ));
    }

    let embed_response = request_handler.process_request(request.into_inner()).await?;
    Ok(Json(embed_response))
}

/// GET /health - Health check endpoint
///
/// Returns "OK" if the service is running.
/// Used by load balancers and monitoring systems.
#[get("/health")]
pub fn health() -> &'static str {
    "OK"
}

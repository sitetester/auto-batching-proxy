use crate::batch_processor::BatchProcessor;
use crate::config::AppConfig;
use crate::types::{EmbedRequest, EmbedResponse, ErrorResponse, PendingRequest};
use rocket::http::Status;
use rocket::response::status::Custom;
use rocket::serde::json::Json;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

pub struct RequestHandler {
    pub config: AppConfig,
    request_sender: mpsc::UnboundedSender<PendingRequest>,
}

impl RequestHandler {
    pub async fn new(config: AppConfig) -> Result<Self, anyhow::Error> {
        // setup mpsc channel
        // - each request will be sent though it, hence `multiple producer`
        // - receiver will be handling requests in tokio spawn`ed task
        let (request_sender, request_receiver): (
            mpsc::UnboundedSender<PendingRequest>,
            mpsc::UnboundedReceiver<PendingRequest>,
        ) = mpsc::unbounded_channel(); // non-blocking

        BatchProcessor::new(&config, request_receiver)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create BatchProcessor: {}", e))?;

        Ok(Self {
            config,
            request_sender,
        })
    }

    pub async fn process_request(
        &self,
        request: EmbedRequest,
    ) -> Result<EmbedResponse, Custom<Json<ErrorResponse>>> {
        // create oneshot channel (only for "this particular" request
        let (response_sender, response_receiver): (
            oneshot::Sender<Result<EmbedResponse, Custom<Json<ErrorResponse>>>>,
            oneshot::Receiver<Result<EmbedResponse, Custom<Json<ErrorResponse>>>>,
        ) = oneshot::channel();

        // inference service supports both single & multiple inputs per user
        let pending_request = PendingRequest::new(request.inputs, response_sender);

        self.request_sender.send(pending_request).map_err(|_| {
            Custom(
                Status::InternalServerError,
                Json(ErrorResponse {
                    error: "Failed to queue request".to_string(),
                }),
            )
        })?;

        // for individual request handling
        // this is different from `--max-wait-time-ms x` which is for our proxy batch execution delay time
        let request_timeout = self.config.max_wait_time_duration() + Duration::from_secs(30);

        timeout(request_timeout, response_receiver)
            .await
            .map_err(|_| {
                Custom(
                    Status::RequestTimeout,
                    Json(ErrorResponse {
                        error: "Request timed out".to_string(),
                    }),
                )
            })?
            .map_err(|_| {
                Custom(
                    Status::InternalServerError,
                    Json(ErrorResponse {
                        error: "Response channel closed".to_string(),
                    }),
                )
            })?
    }
}

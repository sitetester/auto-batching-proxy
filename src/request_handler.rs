use crate::batch_processor::BatchProcessor;
use crate::config::AppConfig;
use crate::inference_client::InferenceServiceClient;
use crate::types::{
    EmbedRequest, EmbedResponse, ErrorResponse, PendingRequest, ResponseReceiver, ResponseSender,
};
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

        // create this client once & return potential error
        let inference_client = InferenceServiceClient::new(&config)
            .map_err(|e| anyhow::anyhow!("Failed to create InferenceServiceClient: {}", e))?;

        let batch_processor = BatchProcessor::new(config.clone(), inference_client);
        // launch `run` as a background task
        tokio::spawn(batch_processor.run(request_receiver));

        Ok(Self {
            config,
            request_sender,
        })
    }

    /// This is further received by `/embed` route
    pub async fn process_request(
        &self,
        request: EmbedRequest,
    ) -> Result<EmbedResponse, Custom<Json<ErrorResponse>>> {
        // create oneshot channel (only for "this particular" request
        let (response_sender, response_receiver): (ResponseSender, ResponseReceiver) =
            oneshot::channel();

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

        // without `timeout`, requests could hang indefinitely, just in case:
        // batch processor gets stuck or downstream inference service becomes unresponsive
        // check ```response_sender.send(Ok(response))``` in batch_processor
        let timeout_result = timeout(request_timeout, response_receiver).await;

        // Result<Result<Result<EmbedResponse, Custom<Json<ErrorResponse>>>, RecvError>, Elapsed>
        let after_timeout_check = timeout_result.map_err(|_| {
            Custom(
                Status::RequestTimeout,
                Json(ErrorResponse {
                    error: "Request timed out".to_string(),
                }),
            )
        })?;
        // => Result<Result<Result<EmbedResponse, Custom<Json<ErrorResponse>>>, RecvError>, Custom<Json<ErrorResponse>>>
        // Result<Result<EmbedResponse, Custom<Json<ErrorResponse>>>, RecvError>
        // (? unwrapped outer layer, early return if timeout)
        after_timeout_check.map_err(|_| {
            Custom(
                Status::InternalServerError,
                Json(ErrorResponse {
                    error: "Response channel closed".to_string(),
                }),
            )
        })?
        // => Result<Result<EmbedResponse, Custom<Json<ErrorResponse>>>, Custom<Json<ErrorResponse>>>
        // Result<EmbedResponse, Custom<Json<ErrorResponse>>>
        // (? unwrapped outer layer, early return if timeout)
        // which is the return type of `process_request(...)`
    }
}

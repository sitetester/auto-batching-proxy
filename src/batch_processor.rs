use crate::config::AppConfig;
use crate::inference_client::{InferenceError, InferenceServiceClient};
use crate::types::{
    BatchInfo, BatchRequest, BatchResponse, BatchType, EmbedResponse, ErrorResponse, PendingRequest,
};
use log::{debug, error, info, warn};
use rocket::response::status::Custom;
use rocket::serde::json::Json;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

pub static BATCH_COUNTER: AtomicU64 = AtomicU64::new(1);

pub struct BatchProcessor {}

impl BatchProcessor {
    pub async fn new(
        config: &AppConfig,
        request_receiver: mpsc::UnboundedReceiver<PendingRequest>,
    ) -> Result<Self, anyhow::Error> {
        // create this client ONCE & return potential error (not possible from inside `tokio::spawn`)
        let inference_client = InferenceServiceClient::new(&config)?;

        // check `RequestHandler::process_request(..)` how such requests are sent via `request_sender`
        tokio::spawn(Self::run_batch_processor(
            config.clone(),
            request_receiver,
            inference_client,
        ));

        Ok(Self {})
    }

    async fn run_batch_processor(
        config: AppConfig,
        mut request_receiver: mpsc::UnboundedReceiver<PendingRequest>,
        inference_client: InferenceServiceClient,
    ) {
        let inference_client = Arc::new(inference_client);
        let mut pending_requests: VecDeque<PendingRequest> = VecDeque::new();
        let mut batch_interval =
            tokio::time::interval(Duration::from_millis(config.batch_check_interval_ms));
        batch_interval.tick().await; // skip the first immediate tick call as it returns immediately (at time 0)

        loop {
            tokio::select! {
                maybe_request = request_receiver.recv() => {
                    if let Some(request) = maybe_request {
                        debug!("Received new request with inputs: {:?}", request.inputs);

                        // `max_inference_inputs` check is applied inside `/embed` route (routes.rs)
                        // & batch size limits are enforced in `build_safe_batch()`
                        pending_requests.push_back(request);

                        if pending_requests.len() >= config.max_batch_size {
                            Self::process_pending_requests(&mut pending_requests, &config, &inference_client,
                                BatchType::MaxBatchSize
                            );
                        }
                    }
                }
                // imagine only 1 request arrived, but then there are no new requests,
                // can cause timeout without even executing `handle_max_wait_time_ms(...)` for older requests,
                // having ticker ensures, this branch runs & eventually processes `handle_max_wait_time_ms(...)`
                _ = batch_interval.tick() => {
                   // periodic wakeup to check pending requests
                }
            }

            // it will reach here, irrespective of which `tokio::select!` branch was picked
            Self::handle_max_wait_time_ms(&mut pending_requests, &config, &inference_client);
        }
    }

    /// ```Max Wait Time - maximal time user request can wait for other requests to be accumulated in a batch```
    ///
    /// let's assume, we have such timeline, at 500th ms, we process all requests in single batch,
    /// (but also consider `max_inference_inputs` limitation)
    ///
    /// User1 request with 10 inputs arrives at 0th ms
    /// User2 request with 20 inputs arrives at 100th ms
    /// User3 request with 10 inputs arrives at 300th ms // exceeds max_inference_inputs of e.g., 32
    /// User4 request with 5 inputs arrives at 500th ms
    fn handle_max_wait_time_ms(
        pending_requests: &mut VecDeque<PendingRequest>,
        config: &AppConfig,
        inference_client: &Arc<InferenceServiceClient>,
    ) {
        if let Some(oldest_request) = pending_requests.front() {
            let received_at = oldest_request.received_at;

            if received_at.elapsed() >= config.max_wait_time_duration() {
                info!(
                    "Processing due to config.max_wait_time_ms: {} timeout",
                    config.max_wait_time_ms
                );
                debug!("oldest request waited {:?}", received_at.elapsed());
                Self::process_pending_requests(
                    pending_requests,
                    config,
                    inference_client,
                    BatchType::MaxWaitTimeMs,
                )
            }
        }
    }

    /// To avoid overwhelming the inference service, it will process in batches
    /// respecting `config.max_batch_size` as well as `config.max_inference_inputs`
    fn process_pending_requests(
        pending_requests: &mut VecDeque<PendingRequest>,
        config: &AppConfig,
        inference_client: &Arc<InferenceServiceClient>,
        batch_type: BatchType,
    ) {
        info!("Processing batch type: {:?}...", batch_type);

        let mut batch_wait_time_ms = Some(config.max_wait_time_ms);
        if batch_type == BatchType::MaxBatchSize {
            // to avoid confusion (whether size or timing), let's not show this info in returned
            // BatchInfo results (in tests), also check ```skip_serializing_if = "Option::is_none"```
            batch_wait_time_ms = None;
        }

        while !pending_requests.is_empty() {
            let batch = Self::build_safe_batch(pending_requests, config);
            if batch.is_empty() {
                debug!("Batch is empty, will skip processing...");
                break;
            }

            let batch_size = batch.len();
            info!("Processing batch size: {}", batch_size);

            let mut batch_info = None;
            if config.include_batch_info {
                batch_info = Some(BatchInfo {
                    batch_id: BATCH_COUNTER.fetch_add(1, Ordering::Relaxed),
                    batch_type,
                    batch_size: Some(batch_size),
                    batch_wait_time_ms,
                    inference_time_ms: None, // filled later in process_batch(...)
                    processing_time_ms: None, // as above
                });
            }

            tokio::spawn(Self::process_batch(
                batch,
                inference_client.clone(),
                batch_info,
            ));
        }
    }

    fn build_safe_batch(
        pending_requests: &mut VecDeque<PendingRequest>,
        config: &AppConfig,
    ) -> Vec<PendingRequest> {
        let mut batch_size = 0;
        let mut inputs_count = 0;

        // `.iter()` - front-to-back iterator
        for request in pending_requests.iter() {
            if batch_size >= config.max_batch_size
                || (inputs_count + request.inputs.len()) > config.max_inference_inputs
            {
                break;
            }
            inputs_count += request.inputs.len();
            batch_size += 1;
        }

        debug!("[build_safe_batch] batch_size: {}", batch_size);
        pending_requests.drain(..batch_size).collect()
    }

    async fn process_batch(
        batch: Vec<PendingRequest>,
        inference_client: Arc<InferenceServiceClient>,
        batch_info: Option<BatchInfo>,
    ) {
        let start_time = Instant::now();
        let batch_response = inference_client
            .call_service(BatchRequest::prepare_request(&batch))
            .await;
        let inference_time_ms = start_time.elapsed();

        match batch_response {
            Ok(embeddings) => {
                Self::handle_batch_success(
                    batch,
                    embeddings,
                    batch_info,
                    start_time,
                    inference_time_ms,
                );
            }
            Err(e) => {
                Self::handle_batch_error(batch, e);
            }
        }
    }

    fn handle_batch_success(
        batch: Vec<PendingRequest>,
        embeddings: BatchResponse,
        mut batch_info: Option<BatchInfo>,
        start_time: Instant,
        inference_time_ms: Duration,
    ) {
        info!(
            "Batch processed successfully in {:?}, {} embeddings returned",
            start_time.elapsed(),
            embeddings.len()
        );

        let mut current_index = 0;
        let batch_size = batch.len();

        if let Some(ref mut info) = batch_info {
            info.inference_time_ms = Some(inference_time_ms.as_millis() as f64);
            info.batch_size = Some(batch_size);
        }

        for pending_request in batch {
            let start_idx = current_index;
            let end_idx = start_idx + pending_request.inputs.len();

            let individual_embeddings = embeddings
                .get(start_idx..end_idx)
                .map(|slice| slice.to_vec())
                .unwrap_or_default();

            if let Some(ref mut info) = batch_info {
                info.processing_time_ms = Some(start_time.elapsed().as_millis() as f64);
            }

            let response = EmbedResponse {
                embeddings: individual_embeddings,
                batch_info: batch_info.clone(),
            };

            // this call is handled by ```timeout(request_timeout, response_receiver).await;``` in process_request(...)
            if pending_request.response_sender.send(Ok(response)).is_err() {
                warn!("Failed to send response to client (may have disconnected)");
            }

            current_index = end_idx;
        }
    }

    fn handle_batch_error(batch: Vec<PendingRequest>, error: InferenceError) {
        error!("Batch processing failed: {:?}", error);

        let error_response = Custom(
            error.to_rocket_status(),
            Json(ErrorResponse {
                error: error.message(),
            }),
        );

        for pending_request in batch {
            if pending_request
                .response_sender
                .send(Err(error_response.clone()))
                .is_err()
            {
                error!("Failed to send error response to client");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::batch_processor::BatchProcessor;
    use crate::config::AppConfig;
    use crate::types::{EmbedResponse, ErrorResponse, PendingRequest};
    use rocket::response::status::Custom;
    use rocket::serde::json::Json;
    use std::collections::VecDeque;
    use tokio::sync::oneshot;
    use tokio::sync::oneshot::Sender;

    type ResponseSender = Sender<Result<EmbedResponse, Custom<Json<ErrorResponse>>>>;

    #[test]
    fn test_build_safe_batch_max_batch_size() {
        let mut config = AppConfig::default();
        config.max_batch_size = 5;

        let mut pending_requests = VecDeque::new();
        for _ in 1..=10 {
            let (response_sender, _): (ResponseSender, _) = oneshot::channel();
            let pending_request = PendingRequest::new(vec!["Hello".to_string()], response_sender);
            pending_requests.push_back(pending_request);
        }

        let result = BatchProcessor::build_safe_batch(&mut pending_requests, &config);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_build_safe_batch_max_inference_inputs() {
        let mut config = AppConfig::default();
        config.max_inference_inputs = 10;

        let inputs: Vec<String> = (1..=5).map(|i| format!("{}: What is NLP", i)).collect();

        let mut pending_requests = VecDeque::new();
        for _ in 1..=3 {
            let (response_sender, _): (ResponseSender, _) = oneshot::channel();
            let pending_request = PendingRequest::new(inputs.clone(), response_sender);
            pending_requests.push_back(pending_request);
        }

        let result = BatchProcessor::build_safe_batch(&mut pending_requests, &config);
        assert_eq!(result.len(), 2);
    }
}

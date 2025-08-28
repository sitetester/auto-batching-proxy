use crate::config::AppConfig;
use rocket::response::status::Custom;
use rocket::serde::json::Json;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;

pub type ResponseSender = oneshot::Sender<Result<EmbedResponse, Custom<Json<ErrorResponse>>>>;
pub type ResponseReceiver = oneshot::Receiver<Result<EmbedResponse, Custom<Json<ErrorResponse>>>>;

#[derive(Serialize, Debug, Clone)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbedRequest {
    /// Inference service supports both single & multiple inputs per user
    pub inputs: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub enum BatchType {
    #[serde(rename = "max_batch_size")]
    MaxBatchSize,
    #[serde(rename = "max_wait_time_ms")]
    MaxWaitTimeMs,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BatchInfo {
    pub batch_id: u64,
    pub batch_type: BatchType,
    pub batch_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_wait_time_ms: Option<u64>,
    pub inference_time_ms: Option<f64>,
}

pub static BATCH_COUNTER: AtomicU64 = AtomicU64::new(1);

impl BatchInfo {
    pub fn new(config: &AppConfig, batch_type: BatchType, batch_size: usize) -> Option<BatchInfo> {
        let batch_wait_time_ms = if batch_type == BatchType::MaxWaitTimeMs {
            Some(config.max_wait_time_ms)
        } else {
            // to avoid confusion (whether size or timing), let's not show this info in returned
            // BatchInfo results in tests, also check ```skip_serializing_if = "Option::is_none"```
            None
        };

        if config.include_batch_info {
            return Some(BatchInfo {
                batch_id: BATCH_COUNTER.fetch_add(1, Ordering::Relaxed),
                batch_type,
                batch_size: Some(batch_size),
                batch_wait_time_ms,
                inference_time_ms: None, // filled later in `process_batch`
            });
        }
        None
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")] // hide when None
    pub batch_info: Option<BatchInfo>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BatchRequest {
    pub inputs: Vec<String>,
}
impl BatchRequest {
    pub fn prepare_request(batch: &[PendingRequest]) -> BatchRequest {
        let all_inputs: Vec<String> = batch
            .iter()
            .flat_map(|request| &request.inputs)
            .cloned()
            .collect();
        BatchRequest { inputs: all_inputs }
    }
}

// TEI returns embeddings directly as an array, not wrapped in an object
pub type BatchResponse = Vec<Vec<f32>>;

#[derive(Debug)]
pub struct PendingRequest {
    pub inputs: Vec<String>,
    pub response_sender: ResponseSender,
    pub received_at: std::time::Instant,
}

impl PendingRequest {
    pub fn new(inputs: Vec<String>, response_sender: ResponseSender) -> Self {
        Self {
            inputs,
            response_sender,
            received_at: std::time::Instant::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use tokio::sync::oneshot;

    #[test]
    fn test_prepare_request_can_handle_duplicates_for_multiple_users() {
        let (response_sender, _response_receiver) = oneshot::channel();
        let req1 = PendingRequest {
            inputs: vec!["Hello".to_string()],
            response_sender,
            received_at: Instant::now(),
        };

        let (response_sender, _response_receiver) = oneshot::channel();
        let req2 = PendingRequest {
            inputs: vec!["Hello".to_string()],
            response_sender,
            received_at: Instant::now(),
        };

        let batch: Vec<PendingRequest> = vec![req1, req2];
        let prepared = BatchRequest::prepare_request(&batch);

        assert_eq!(prepared.inputs.len(), 2);
        assert_eq!(prepared.inputs[0], "Hello");
        assert_eq!(prepared.inputs[1], "Hello");
    }

    #[test]
    fn test_prepare_request_can_handle_multiple_inputs_per_user() {
        let (response_sender, _response_receiver) = oneshot::channel();
        let req = PendingRequest {
            inputs: vec!["Hello".to_string(), "World".to_string()],
            response_sender,
            received_at: Instant::now(),
        };

        let batch: Vec<PendingRequest> = vec![req];
        let prepared = BatchRequest::prepare_request(&batch);

        assert_eq!(prepared.inputs.len(), 2);
        assert_eq!(prepared.inputs[0], "Hello");
        assert_eq!(prepared.inputs[1], "World");
    }
}

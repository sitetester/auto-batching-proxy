use crate::config::AppConfig;
use crate::types::{BatchRequest, BatchResponse};
use anyhow::{Result, anyhow};
use log::debug;
use reqwest::Error;
use std::time::Duration;

#[derive(Debug)]
pub enum InferenceError {
    NetworkError(Error),
    HttpError {
        status: reqwest::StatusCode,
        body: String,
    },
    ParseError(Error),
}
impl InferenceError {
    pub fn to_rocket_status(&self) -> rocket::http::Status {
        match self {
            InferenceError::NetworkError(_) => rocket::http::Status::ServiceUnavailable,
            InferenceError::HttpError { status, .. } => match status.as_u16() {
                400..=499 => rocket::http::Status::BadRequest,
                500..=599 => rocket::http::Status::ServiceUnavailable,
                _ => rocket::http::Status::InternalServerError,
            },
            InferenceError::ParseError(_) => rocket::http::Status::InternalServerError,
        }
    }

    pub fn message(&self) -> String {
        match self {
            InferenceError::NetworkError(e) => format!("Network error: {}", e),
            InferenceError::HttpError { status, body } => {
                format!("HTTP {}: {}", status, body)
            }
            InferenceError::ParseError(e) => format!("Parse error: {}", e),
        }
    }
}

#[derive(Clone)]
pub struct InferenceServiceClient {
    client: reqwest::Client,
    base_url: String,
}

impl InferenceServiceClient {
    pub fn new(config: &AppConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.inference_timeout_secs))
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        Ok(Self {
            client,
            base_url: config.inference_url.clone(),
        })
    }

    pub async fn call_service(
        &self,
        request: BatchRequest,
    ) -> Result<BatchResponse, InferenceError> {
        debug!(
            "Making request to inference service: {} with {} inputs: {:?}",
            self.base_url,
            request.inputs.len(),
            request.inputs
        );

        let response = self
            .client
            .post(&self.base_url)
            .json(&request)
            .send()
            .await
            .map_err(InferenceError::NetworkError)?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(InferenceError::HttpError { status, body });
        }

        let batch_response: BatchResponse =
            response.json().await.map_err(InferenceError::ParseError)?;

        Ok(batch_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;

    #[test]
    fn test_new_success() {
        let config = AppConfig::default();
        let result = InferenceServiceClient::new(&config);
        assert_eq!(result.unwrap().base_url, config.inference_url.to_string());
    }

    #[tokio::test]
    async fn test_call_service_success() {
        let config = AppConfig::default();
        let result = InferenceServiceClient::new(&config);
        let client = result.unwrap();
        let request = BatchRequest {
            inputs: vec!["hello".to_string(), "world".to_string()],
        };
        let response = client.call_service(request).await;
        assert_eq!(response.unwrap().len(), 2);
    }
}
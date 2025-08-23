use auto_batching_proxy::types::BatchType;
use auto_batching_proxy::{build_rocket, config::AppConfig};
use rocket::http::ContentType;
use rocket::local::asynchronous::{Client, LocalResponse};
use serde_json::{Value, json};
use std::sync::Arc;

pub async fn get_client(config: AppConfig) -> Client {
    let rocket = build_rocket(config).await;
    Client::tracked(rocket)
        .await
        .expect("valid rocket instance")
}

pub async fn get_client_with_defaults() -> Client {
    let config = AppConfig::default();
    let rocket = build_rocket(config).await;
    Client::tracked(rocket)
        .await
        .expect("valid rocket instance")
}

/// Helper function to make POST requests with JSON body using provided client
pub async fn post_json<'a>(
    client: &'a Client,
    route: &'a str,
    json_body: String,
) -> LocalResponse<'a> {
    client
        .post(route)
        .header(ContentType::JSON)
        .body(json_body)
        .dispatch()
        .await
}

/// CAUTION! - inference service supports max 32 inputs
pub async fn launch_threads_with_tests(
    client: Arc<Client>,
    num: usize,
    inputs: Arc<Vec<String>>,
    skip_tests: bool,
) -> Vec<Value> {
    let mut handles = Vec::new();
    for _ in 1..=num {
        let client = client.clone();
        let inputs = inputs.clone();
        let handle = tokio::spawn(async move {
            let response = post_json(
                &*client, // dereference Arc to get &Client
                "/embed",
                json!({"inputs": *inputs}).to_string(),
            )
            .await;

            let json: Value = response.into_json().await.expect("Valid JSON");
            if skip_tests {
                assert!(
                    json["embeddings"].is_array(),
                    "Response should contain embeddings array"
                );

                let embeddings = json["embeddings"].as_array().unwrap();
                assert_eq!(embeddings.len(), inputs.len(),);

                assert!(
                    embeddings[0].is_array(),
                    "Each embedding should be an array of numbers"
                );

                let embedding_values = embeddings[0].as_array().unwrap();
                assert!(
                    !embedding_values.is_empty(),
                    "Embedding should not be empty"
                );

                for value in embedding_values {
                    assert!(value.is_number(), "All embedding values should be numbers");
                }
            }
            json["batching_info"].clone()
        });
        handles.push(handle);
    }

    let mut results = Vec::new();
    for h in handles {
        results.push(h.await.unwrap());
    }
    results
}

pub fn build_inputs(num: usize, mut maybe_input: Option<&str>) -> Vec<String> {
    let input = maybe_input.get_or_insert("What is Vector search ?");

    let inputs: Vec<String> = if num == 1 {
        vec![input.to_string()]
    } else {
        (1..=num).map(|i| format!("{}: {}", i, input)).collect()
    };

    inputs
}

pub async fn direct_call_to_inference_service(inputs: &Vec<String>) -> Vec<Vec<f32>> {
    let inference_client = reqwest::Client::new();
    let direct_response = inference_client
        .post(&AppConfig::default().inference_url)
        .header("Content-Type", "application/json")
        .json(&json!({
            "inputs": inputs
        }))
        .send()
        .await
        .expect("Direct inference call should succeed");

    let direct_embeddings: Vec<Vec<f32>> = direct_response
        .json()
        .await
        .expect("Should parse direct response");

    direct_embeddings
}

pub fn batch_type_and_size(results: &Vec<Value>, batch_type: BatchType, size: usize) -> usize {
    results
        .iter()
        .filter(|info| {
            // deserialize the JSON value to BatchType
            let json_batch_type: Result<BatchType, _> =
                serde_json::from_value(info["batch_type"].clone());
            // `false` will make the `filter` fail
            json_batch_type.map_or(false, |bt| bt == batch_type) && info["batch_size"] == size
        })
        .count()
}

pub fn get_proxy_embeddings(json: Value) -> Vec<Vec<f32>> {
    json["embeddings"]
        .as_array() // Get the embeddings array
        .expect("embeddings should be an array")
        .iter()
        .map(|embedding| {
            // For each embedding
            embedding
                .as_array() // Convert to array of numbers
                .expect("each embedding should be an array")
                .iter()
                .map(|val| val.as_f64().expect("embedding values should be numbers") as f32)
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>()
}

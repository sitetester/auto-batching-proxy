#![allow(dead_code)] // for some reason, it's generating warnings for used functions

use auto_batching_proxy::types::{BatchInfo, BatchType};
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

/// Helper function to make POST requests with JSON body using Rocket's internal test client
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

/// CAUTION! - inference service could have max inputs limit like 32
pub async fn launch_threads_with_tests(
    client: Arc<Client>,
    num: usize,
    inputs: Vec<String>,
    run_assertions: bool,
) -> Vec<Value> {
    let mut handles = Vec::new();
    let inputs = Arc::new(inputs);
    for _ in 1..=num {
        let client = client.clone();
        let inputs = inputs.clone();
        let handle = tokio::spawn(async move {
            let response = post_json(
                client.as_ref(), // alternatively &*client, as * causes Deref
                "/embed",
                json!({"inputs": *inputs}).to_string(),
            )
            .await;

            let json: Value = response.into_json().await.expect("Valid JSON");
            if run_assertions {
                assert!(
                    json["embeddings"].is_array(),
                    "Response should contain embeddings array"
                );

                let embeddings = json["embeddings"].as_array().unwrap();
                assert_eq!(embeddings.len(), inputs.len(),);

                let mut first_embedding_len = 0;
                for (i, embedding) in embeddings.iter().enumerate() {
                    assert!(
                        embedding.is_array(),
                        "Embedding {i} should be an array"
                    );

                    let embedding_values = embedding.as_array().unwrap();
                    assert!(
                        !embedding_values.is_empty(),
                        "Embedding should not be empty"
                    );

                    if i == 0 {
                        // let's define it here (not outside the loop, as then it could fail)
                        // here, it's safe to access such length after prior asserts
                        first_embedding_len = embedding_values.len();
                    }

                    if i > 0 {
                        assert_eq!(
                            embedding_values.len(),
                            first_embedding_len,
                            "All embeddings should have equal length"
                        );
                    }

                    for value in embedding_values {
                        assert!(value.is_number(), "All embedding values should be numbers");
                    }
                }
            }
            // it is assumed `batch_info` is ALWAYS included while running tests (config.include_batch_info = true)
            json["batch_info"].clone()
        });
        handles.push(handle);
    }

    let mut batches_info = Vec::new();
    for h in handles {
        batches_info.push(h.await.unwrap());
    }
    batches_info
}

pub fn build_inputs(num: usize, mut maybe_input: Option<&str>) -> Vec<String> {
    let input = maybe_input.get_or_insert("What is Vector search ?");

    let inputs: Vec<String> = if num == 1 {
        vec![input.to_string()]
    } else {
        (1..=num).map(|i| format!("{i}: {input}")).collect()
    };

    inputs
}

pub async fn direct_call_to_inference_service(inputs: &Vec<String>) -> Vec<Vec<f32>> {
    // compare this with `post_json` which uses Rocket test client
    let inference_client = reqwest::Client::new();
    let response = inference_client
        .post(&AppConfig::default().inference_url) // bypasses our proxy
        .header("Content-Type", "application/json")
        .json(&json!({
            "inputs": inputs
        }))
        .send()
        .await
        .expect("Direct inference call should succeed");

    let embeddings: Vec<Vec<f32>> = response.json().await.expect("Should parse direct response");
    embeddings
}

pub fn count_batch(batches_info: &Vec<Value>, batch_type: BatchType, size: usize) -> usize {
    batches_info
        .iter()
        .filter(|batch_info| {
            // deserialize the JSON value to BatchType
            let batch_info_result: Result<BatchInfo, _> =
                serde_json::from_value((*batch_info).clone());

            match batch_info_result {
                // convert to bool, `false` will make the `filter` fail
                Ok(batch_info) => {
                    batch_info.batch_type == batch_type && batch_info.batch_size == Some(size)
                }
                Err(_) => false,
            }
        })
        .count()
}

pub fn get_proxy_embeddings(json: Value) -> Vec<Vec<f32>> {
    let proxy_embeddings: Vec<Vec<f32>> =
        serde_json::from_value(json["embeddings"].clone()).expect("Should parse embeddings");
    proxy_embeddings
}

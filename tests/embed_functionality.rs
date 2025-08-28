mod test_utils;

use crate::test_utils::{
    build_inputs, count_batch, direct_call_to_inference_service, get_client,
    get_client_with_defaults, launch_threads_with_tests, post_json,
};
use auto_batching_proxy::config::AppConfig;
use auto_batching_proxy::types::BatchType;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

async fn success_with_num_inputs_using_defaults(num: usize) {
    let client = get_client_with_defaults().await;
    let response = post_json(
        &client,
        "/embed",
        json!({
            "inputs":build_inputs(num, None),
        })
        .to_string(),
    )
    .await;

    let json: Value = response.into_json().await.expect("Valid JSON response");

    // this "index" syntax works because of Index trait implementation
    assert!(json["embeddings"].is_array());
    assert_eq!(json["embeddings"].as_array().unwrap().len(), num);
}

#[tokio::test]
async fn test_embed_endpoint_success_single_input_using_defaults() {
    success_with_num_inputs_using_defaults(1).await
}

#[tokio::test]
async fn test_embed_endpoint_success_multiple_inputs_using_defaults() {
    success_with_num_inputs_using_defaults(3).await
}

// --- max_batch_size - start ---
// here we assume, inputs are within `config.max_inference_inputs = 32` range
async fn max_batch_size_should_process_first_with_num_inputs_per_request(num: usize) {
    let config = AppConfig {
        include_batch_info: true,
        max_batch_size: 5, // smaller value
        max_wait_time_ms: 1000,
        ..Default::default()
    };

    let client = Arc::new(get_client(config).await);
    let batches_info =
        launch_threads_with_tests(client.clone(), 7, Arc::new(build_inputs(num, None)), true).await;
    assert_eq!(batches_info.len(), 7);

    assert_eq!(count_batch(&batches_info, BatchType::MaxBatchSize, 5,), 5); // first batch
    assert_eq!(count_batch(&batches_info, BatchType::MaxWaitTimeMs, 2), 2); // second batch
}
#[tokio::test]
async fn test_embed_endpoint_max_batch_size_should_process_first_with_single_input_per_request() {
    max_batch_size_should_process_first_with_num_inputs_per_request(1).await;
}

#[tokio::test]
async fn test_embed_endpoint_max_batch_size_should_process_first_with_multiple_inputs_per_request()
{
    max_batch_size_should_process_first_with_num_inputs_per_request(3).await;
}

#[tokio::test]
async fn test_embed_endpoint_max_batch_size_while_exceeding_max_inference_inputs() {
    let config = AppConfig {
        max_inference_inputs: 32,
        include_batch_info: true,
        max_batch_size: 4, // smaller value, max 4 requests per batch
        max_wait_time_ms: 1000,
        ..Default::default()
    };

    let client = Arc::new(get_client(config).await);
    // launch 7 requests, each with 10 inputs, total 7 * 10 = 70 inputs
    let batches_info =
        launch_threads_with_tests(client.clone(), 7, Arc::new(build_inputs(10, None)), true).await;
    assert_eq!(batches_info.len(), 7);

    // hence, these will be split into 3 batches respecting `config.max_inference_inputs`
    // `max_batch_size = 4` will be triggered, since total launched requests are 7

    // first batch will serve 3 requests (3 * 10 = 30 inputs with BatchType::MaxBatchSize)
    // (because adding 4th request will make it 4 * 10 = 40 which is > 32
    assert_eq!(count_batch(&batches_info, BatchType::MaxBatchSize, 3), 3);
    // second batch will serve 1 request (1 * 10 = 10 inputs with BatchType::MaxBatchSize)
    assert_eq!(count_batch(&batches_info, BatchType::MaxBatchSize, 1), 1);

    // third batch will serve 3 remaining requests (3 * 10 = 30 inputs with BatchType::MaxWaitTimeMs)
    assert_eq!(count_batch(&batches_info, BatchType::MaxWaitTimeMs, 3), 3);
}
// max_batch_size - end

// --- max_wait_time_ms - start ---
// here we assume, inputs are within `config.max_inference_inputs = 32` range
async fn max_wait_time_ms_should_process_first_with_num_inputs_per_request(num: usize) {
    let config = AppConfig {
        include_batch_info: true,
        max_batch_size: 100,
        max_wait_time_ms: 500, // smaller value
        ..Default::default()
    };

    let client = Arc::new(get_client(config).await);
    let batches_info =
        launch_threads_with_tests(client, 3, Arc::new(build_inputs(num, None)), true).await;
    assert_eq!(batches_info.len(), 3);

    assert_eq!(count_batch(&batches_info, BatchType::MaxWaitTimeMs, 3), 3);
}

#[tokio::test]
async fn test_embed_endpoint_success_max_wait_time_ms_should_process_first_with_single_input_per_request()
 {
    max_wait_time_ms_should_process_first_with_num_inputs_per_request(1).await;
}

#[tokio::test]
async fn test_embed_endpoint_success_max_wait_time_ms_should_process_first_with_multiple_input_per_request()
 {
    max_wait_time_ms_should_process_first_with_num_inputs_per_request(5).await;
}

#[tokio::test]
async fn test_embed_endpoint_max_wait_time_ms_while_exceeding_max_inference_inputs() {
    let config = AppConfig {
        include_batch_info: true,
        max_batch_size: 100,
        max_wait_time_ms: 500, // smaller value
        ..Default::default()
    };

    let client = Arc::new(get_client(config).await);
    let batches_info =
        launch_threads_with_tests(client.clone(), 7, Arc::new(build_inputs(10, None)), true).await;
    assert_eq!(batches_info.len(), 7);

    assert_eq!(count_batch(&batches_info, BatchType::MaxWaitTimeMs, 3), 6); // first & second batch
    assert_eq!(count_batch(&batches_info, BatchType::MaxWaitTimeMs, 1), 1); // third batch
}
// max_wait_time_ms - end

#[tokio::test]
async fn test_compare_single_input_inference_service_vs_auto_batching_proxy_with_x_separate_requests()
 {
    let config = AppConfig {
        // this will make Rocket silent :) (check lib.rs)
        quiet_mode: true,
        // let's not return any debug stuff in API responses, since our focus is on performance for now
        include_batch_info: false,
        max_batch_size: 30,
        max_wait_time_ms: 50,
        ..Default::default()
    };

    let client = Arc::new(get_client(config).await);
    let requests = [1, 5, 10, 25, 30, 50, 75, 100, 200, 500, 1000];

    // test different request counts
    let mut direct_timings: BTreeMap<usize, Duration> = BTreeMap::new();
    for &num_requests in &requests {
        let start_time = std::time::Instant::now();
        for _ in 1..=num_requests {
            direct_call_to_inference_service(&build_inputs(1, None)).await;
        }
        direct_timings.insert(num_requests, start_time.elapsed());
    }

    // proxy
    let mut proxy_timings: BTreeMap<usize, Duration> = BTreeMap::new();
    for &num_requests in &requests {
        let start_time = std::time::Instant::now();
        launch_threads_with_tests(
            client.clone(),
            num_requests,
            Arc::new(build_inputs(1, None)),
            false,
        )
        .await;
        proxy_timings.insert(num_requests, start_time.elapsed());
    }

    let print_timing = |proxy_timings| {
        println!("Requests {:^20} {:^15}", "Direct", "Proxy");
        for (num_requests, proxy_elapsed) in proxy_timings {
            println!(
                "{:^5} {:^25} {:^10}",
                num_requests,
                direct_timings
                    .get(&num_requests)
                    .map_or("N/A".to_string(), |elapsed| format!("{elapsed:?}")),
                format!("{:?}", proxy_elapsed)
            );
        }
    };

    println!("\nTiming Summary:");
    print_timing(proxy_timings);
}

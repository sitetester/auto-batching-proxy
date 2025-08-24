mod test_utils;

use crate::test_utils::{
    batch_type_and_size, build_inputs, direct_call_to_inference_service, get_client,
    get_client_with_defaults, launch_threads_with_tests, post_json,
};
use auto_batching_proxy::config::AppConfig;
use auto_batching_proxy::types::BatchType;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn test_embed_endpoint_success_single_input_using_defaults() {
    let client = get_client_with_defaults().await;
    let response = post_json(
        &client,
        "/embed",
        json!({
            "inputs": build_inputs(1, None),
        })
        .to_string(),
    )
    .await;

    let json: Value = response.into_json().await.expect("Valid JSON response");

    // this "index" syntax works because of Index trait implementation
    assert!(json["embeddings"].is_array());
    assert_eq!(json["embeddings"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn test_embed_endpoint_success_multiple_inputs_using_defaults() {
    let num: usize = 3;
    let client = get_client_with_defaults().await;
    let response = post_json(
        &client,
        "/embed",
        json!({
            "inputs": build_inputs(num, None),
        })
        .to_string(),
    )
    .await;

    let json: Value = response.into_json().await.expect("Valid JSON response");
    assert!(json["embeddings"].is_array());
    assert_eq!(json["embeddings"].as_array().unwrap().len(), num);
}

// max_batch_size - start
#[tokio::test]
async fn test_embed_endpoint_max_batch_size_should_process_first_with_single_input_per_request() {
    let mut config = AppConfig::default();
    config.include_batch_info = true;
    config.max_batch_size = 5;
    config.max_wait_time_ms = 1000;

    let client = Arc::new(get_client(config).await);
    let batches_info =
        launch_threads_with_tests(client.clone(), 7, Arc::new(build_inputs(1, None)), true).await;
    assert_eq!(batches_info.len(), 7);

    assert_eq!(
        batch_type_and_size(&batches_info, BatchType::MaxBatchSize, 5,),
        5
    ); // first batch
    assert_eq!(
        batch_type_and_size(&batches_info, BatchType::MaxWaitTimeMs, 2),
        2
    ); // second batch
}

#[tokio::test]
async fn test_embed_endpoint_max_batch_size_should_process_first_with_multiple_inputs_per_request()
{
    let mut config = AppConfig::default();
    config.include_batch_info = true;
    config.max_batch_size = 5;
    config.max_wait_time_ms = 1000;

    let client = Arc::new(get_client(config).await);
    let batches_info =
        launch_threads_with_tests(client.clone(), 7, Arc::new(build_inputs(3, None)), true).await;
    assert_eq!(batches_info.len(), 7);

    assert_eq!(
        batch_type_and_size(&batches_info, BatchType::MaxBatchSize, 5),
        5
    ); // first batch
    assert_eq!(
        batch_type_and_size(&batches_info, BatchType::MaxWaitTimeMs, 2),
        2
    ); // second batch
}

#[tokio::test]
async fn test_embed_endpoint_max_batch_size_while_exceeding_max_inference_inputs() {
    let mut config = AppConfig::default();
    config.include_batch_info = true;
    config.max_batch_size = 4;
    config.max_wait_time_ms = 1000;

    let client = Arc::new(get_client(config).await);
    let batches_info =
        launch_threads_with_tests(client.clone(), 7, Arc::new(build_inputs(10, None)), true).await;
    assert_eq!(batches_info.len(), 7);

    // max_batch_size = 4 covered with splits to config.max_inference_inputs
    assert_eq!(
        batch_type_and_size(&batches_info, BatchType::MaxBatchSize, 3),
        3
    ); // first batch
    assert_eq!(
        batch_type_and_size(&batches_info, BatchType::MaxBatchSize, 1),
        1
    ); // second batch

    assert_eq!(
        batch_type_and_size(&batches_info, BatchType::MaxWaitTimeMs, 3),
        3
    ); // third batch
}
// max_batch_size - end

// max_wait_time_ms - start
#[tokio::test]
async fn test_embed_endpoint_success_max_wait_time_ms_should_process_first_with_single_input_per_request()
 {
    let mut config = AppConfig::default();
    config.include_batch_info = true;
    config.max_batch_size = 100;
    config.max_wait_time_ms = 500;

    let client = Arc::new(get_client(config).await);
    let batches_info =
        launch_threads_with_tests(client, 3, Arc::new(build_inputs(1, None)), true).await;
    assert_eq!(batches_info.len(), 3);

    assert_eq!(
        batch_type_and_size(&batches_info, BatchType::MaxWaitTimeMs, 3),
        3
    );
}

#[tokio::test]
async fn test_embed_endpoint_success_max_wait_time_ms_should_process_first_with_multiple_input_per_request()
 {
    let mut config = AppConfig::default();
    config.include_batch_info = true;
    config.max_batch_size = 100;
    config.max_wait_time_ms = 500;

    let client = Arc::new(get_client(config).await);
    let batches_info =
        launch_threads_with_tests(client, 3, Arc::new(build_inputs(5, None)), true).await;
    assert_eq!(batches_info.len(), 3);

    assert_eq!(
        batch_type_and_size(&batches_info, BatchType::MaxWaitTimeMs, 3),
        3
    );
}

#[tokio::test]
async fn test_embed_endpoint_max_wait_time_ms_while_exceeding_max_inference_inputs() {
    let mut config = AppConfig::default();
    config.include_batch_info = true;
    config.max_batch_size = 100;
    config.max_wait_time_ms = 500;

    let client = Arc::new(get_client(config).await);
    let batches_info =
        launch_threads_with_tests(client.clone(), 7, Arc::new(build_inputs(10, None)), true).await;
    assert_eq!(batches_info.len(), 7);

    assert_eq!(
        batch_type_and_size(&batches_info, BatchType::MaxWaitTimeMs, 3),
        6
    ); // first & second batch
    assert_eq!(
        batch_type_and_size(&batches_info, BatchType::MaxWaitTimeMs, 1),
        1
    ); // third batch
}
// max_wait_time_ms - end

#[tokio::test]
async fn test_compare_single_input_inference_service_vs_auto_batching_proxy_with_30_separate_requests()
 {
    let mut config = AppConfig::default();
    config.include_batch_info = true;
    config.max_batch_size = 30;
    config.max_wait_time_ms = 50;

    let client = Arc::new(get_client(config).await);

    // test different request counts
    let mut direct_timings: BTreeMap<usize, Duration> = BTreeMap::new();
    let requests = [1, 5, 10, 25, 30, 50];
    for &num_requests in &requests {
        let start_time = std::time::Instant::now();
        for _ in 1..=num_requests {
            direct_call_to_inference_service(&build_inputs(1, None)).await;
        }
        direct_timings.insert(num_requests, start_time.elapsed());
    }

    // proxy
    let mut proxy_timings: BTreeMap<usize, Duration> = BTreeMap::new();
    let requests = [1_usize, 5, 10, 25, 30, 50, 75, 100];
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
        println!("{} {:^20} {:^15}", "Requests", "Direct", "Proxy");
        for (num_requests, proxy_elapsed) in proxy_timings {
            println!(
                "{:^5} {:^25} {:^10}",
                num_requests,
                direct_timings
                    .get(&num_requests)
                    .map_or("N/A".to_string(), |elapsed| format!("{:?}", elapsed)),
                format!("{:?}", proxy_elapsed)
            );
        }
    };

    println!("\nTiming Summary:");
    print_timing(proxy_timings);
}

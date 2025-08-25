mod test_utils;

use crate::test_utils::{
    build_inputs, direct_call_to_inference_service, get_client, get_client_with_defaults,
    get_proxy_embeddings, post_json,
};
use auto_batching_proxy::config::AppConfig;
use rocket::http::{ContentType, Status};
use serde_json::{Value, json};

#[tokio::test]
async fn test_embed_endpoint_plain_text_request() {
    let client = get_client_with_defaults().await;
    let response = client
        .post("embed")
        .header(ContentType::Text) // Wrong content type
        .body("blah")
        .dispatch()
        .await;
    assert_eq!(response.status(), Status::BadRequest);
}

#[tokio::test]
async fn test_embed_endpoint_no_inputs_field() {
    let client = get_client_with_defaults().await;
    let response = post_json(&client, "/embed", json!({}).to_string()).await;
    assert_eq!(response.status(), Status::UnprocessableEntity);
}

#[tokio::test]
async fn test_embed_endpoint_empty_inputs() {
    let client = get_client_with_defaults().await;
    let response = post_json(
        &client,
        "/embed",
        json!({
            "inputs": []
        })
        .to_string(),
    )
    .await;
    assert_eq!(response.status(), Status::BadRequest);
}

#[tokio::test]
async fn test_embed_endpoint_fails_when_inputs_exceed_config_max_inference_inputs() {
    let mut config = AppConfig::default();
    config.max_inference_inputs = 20;

    let inputs = build_inputs(25, None);
    let client = get_client(config).await;
    let response = post_json(
        &client,
        "/embed",
        json!({
            "inputs": inputs
        })
        .to_string(),
    )
    .await;

    assert_eq!(response.status(), Status::PayloadTooLarge);

    let body: Value = response.into_json().await.expect("Valid JSON");
    assert!(body.is_object());
    assert!(body["error"].is_string());
    assert_eq!(body["error"], "`inputs` can't be greater than 20");
    // inference service returns `413 Payload Too Large error!`
}

#[tokio::test]
async fn test_embed_endpoint_succeeds_when_inputs_equals_config_max_inference_inputs() {
    // let's try with defaults this time
    let inputs = build_inputs(AppConfig::default().max_inference_inputs, None);
    let client = get_client_with_defaults().await;
    let response = post_json(
        &client,
        "/embed",
        json!({
            "inputs": inputs
        })
        .to_string(),
    )
    .await;

    assert_eq!(response.status(), Status::Ok);
    // skip the embeddings part this time, checked somewhere else
}

#[tokio::test]
async fn test_embed_endpoint_invalid_json_plain_text() {
    let client = get_client_with_defaults().await;
    let response = post_json(&client, "/embed", "dummy plain text".to_string()).await;
    assert_eq!(response.status(), Status::BadRequest);

    let body: Value = response.into_json().await.expect("Valid JSON");

    println!("{:?}", body);
}

#[tokio::test]
async fn test_embed_endpoint_invalid_json_missing_quotes() {
    let client = get_client_with_defaults().await;
    let invalid_json = r#"{"inputs": ["dummy plain text}"#;
    let response = post_json(&client, "/embed", invalid_json.to_string()).await;
    assert_eq!(response.status(), Status::BadRequest);

    let body: Value = response.into_json().await.expect("Valid JSON response");
    println!("Invalid JSON error: {:?}", body);
}

#[tokio::test]
async fn test_404_not_found() {
    let client = get_client_with_defaults().await;
    let response = client.get("/nonexistent").dispatch().await;
    assert_eq!(response.status(), Status::NotFound);
}

#[tokio::test]
async fn test_verify_direct_and_proxy_return_similar_results_for_single_input() {
    let inputs = vec!["What is ML ?".to_string()];
    let direct_embeddings = direct_call_to_inference_service(&inputs).await;

    let client = get_client_with_defaults().await;
    let response = post_json(
        &client,
        "/embed",
        json!({
            "inputs": inputs,
        })
        .to_string(),
    )
    .await;

    let json: Value = response.into_json().await.expect("Valid JSON response");
    let proxy_embeddings = get_proxy_embeddings(json);

    // this could potentially fail due to floating point comparison
    // test passing on ` --model-id sentence-transformers/all-MiniLM-L6-v2`
    assert_eq!(direct_embeddings, proxy_embeddings);
    // safe alternative
    assert_eq!(direct_embeddings.len(), proxy_embeddings.len());
}

#[tokio::test]
async fn test_verify_direct_and_proxy_return_similar_results_for_2_inputs() {
    let inputs = vec!["What is ML ?".to_string(), "What is NLP ?".to_string()];
    let direct_embeddings = direct_call_to_inference_service(&inputs).await;

    let client = get_client_with_defaults().await;
    let response = post_json(
        &client,
        "/embed",
        json!({
            "inputs": inputs,
        })
        .to_string(),
    )
    .await;

    let json: Value = response.into_json().await.expect("Valid JSON response");
    let proxy_embeddings = get_proxy_embeddings(json);

    // this could potentially fail due to floating point comparison
    // test passing on ` --model-id sentence-transformers/all-MiniLM-L6-v2`
    assert_eq!(direct_embeddings, proxy_embeddings);
    // safe alternative
    assert_eq!(direct_embeddings.len(), proxy_embeddings.len());
}

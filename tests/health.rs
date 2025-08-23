mod test_utils;

use rocket::http::Status;
use test_utils::get_client_with_defaults;

#[tokio::test]
async fn test_health_endpoint() {
    let client = get_client_with_defaults().await;
    let response = client.get("/health").dispatch().await;
    assert_eq!(response.status(), Status::Ok);

    let body = response.into_string().await.expect("valid response body");
    assert_eq!(body, "OK");
}
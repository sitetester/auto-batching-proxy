use auto_batching_proxy::{
    build_rocket,
    config::{AppConfig, Args},
};
use clap::Parser;
use rocket::{launch, Build, Rocket};

#[launch]
async fn rocket() -> Rocket<Build> {
    let args = Args::parse();
    let config = AppConfig::build(Some(args));

    // Initialize logging and get effective log level
    let effective_log_level = config.init_logging();

    log::info!("🚀 Starting auto-batching proxy server...");

    println!(
        "Configuration:\n\tmax_wait_time_ms: {}\n\tmax_batch_size: {}\n\tinference_url: {}\n\tlog_level: {}",
        config.max_wait_time_ms, config.max_batch_size, config.inference_url, effective_log_level
    );

    build_rocket(config).await
}

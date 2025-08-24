use auto_batching_proxy::{
    build_rocket,
    config::{AppConfig, Args},
};
use clap::Parser;
use log::{info};
use rocket::{launch, Build, Rocket};

#[launch]
async fn rocket() -> Rocket<Build> {
    let args = Args::parse();
    let config = AppConfig::build(Some(args));

    // Initialize logging and get effective log level
    let _effective_log_level = config.init_logging();

    info!("🚀 Starting auto-batching proxy server...");
    println!("Server Configuration:");
    println!("  Port: {}", config.port);
    println!("  Batch Settings:");
    println!("    Max batch size: {}", config.max_batch_size);
    println!("    Max wait time: {}ms", config.max_wait_time_ms);
    println!("    Check interval: {}ms", config.batch_check_interval_ms);
    println!("  Inference:");
    println!("    URL: {}", config.inference_url);
    println!("    Timeout: {}s", config.inference_timeout_secs);
    println!("    Max inputs per request: {}", config.max_inference_inputs);
    println!("  Options:");
    println!("    Include batch info: {}", config.include_batch_info);
    println!("    Log level: {}", config.log_level);
    println!();


    build_rocket(config).await
}

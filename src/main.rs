use auto_batching_proxy::{
    build_rocket,
    config::{AppConfig, Args},
};
use clap::Parser;
use log::info;
use rocket::{Build, Rocket, launch};

#[launch]
async fn rocket() -> Rocket<Build> {
    let args = Args::parse();
    let config = AppConfig::build(Some(args)).unwrap_or_else(|err| {
        eprintln!("Configuration error: {err}");
        std::process::exit(1);
    });

    // Initialize logging and get effective log level
    let _effective_log_level = config.init_logging();

    info!("🚀 Starting auto-batching proxy server...");

    println!("Server Configuration:");
    println!("  port: {}", config.port);
    println!("  Batch Settings:");
    println!("    max_batch_size: {}", config.max_batch_size);
    println!("    max_wait_time_ms: {}", config.max_wait_time_ms);
    println!(
        "    batch_check_interval_ms: {}",
        config.batch_check_interval_ms
    );
    println!("  Inference:");
    println!("    inference_url: {}", config.inference_url);
    println!(
        "    inference_timeout_secs: {}",
        config.inference_timeout_secs
    );
    println!("    max_inference_inputs: {}", config.max_inference_inputs);
    println!("  Options:");
    println!("    include_batch_info: {}", config.include_batch_info);
    println!("    log_level: {}", config.log_level);
    println!("    quiet_mode: {}", config.quiet_mode);
    println!();

    build_rocket(config).await
}

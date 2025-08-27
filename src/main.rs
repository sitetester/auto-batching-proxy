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
        println!("Configuration error: {err:?}");
        std::process::exit(1);
    });

    // Initialize logging and get effective log level
    let _effective_log_level = config.init_logging();

    info!("🚀 Starting auto-batching proxy server...");

    // single print syscall
    println!(
        "Server Configuration:
  port: {}
  Batch Settings:
    max_batch_size: {}
    max_wait_time_ms: {}
    batch_check_interval_ms: {}
  Inference:
    inference_url: {}
    inference_timeout_secs: {}
    max_inference_inputs: {}
  Options:
    include_batch_info: {}
    log_level: {}
    quiet_mode: {}
",
        config.port,
        //
        config.max_batch_size,
        config.max_wait_time_ms,
        config.batch_check_interval_ms,
        //
        config.inference_url,
        config.inference_timeout_secs,
        config.max_inference_inputs,
        //
        config.include_batch_info,
        config.log_level,
        config.quiet_mode
    );

    build_rocket(config).await
}

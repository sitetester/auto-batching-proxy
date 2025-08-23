use clap::Parser;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Rocket server port to run the proxy on
    #[arg(long)]
    pub port: Option<u16>,

    /// Maximum wait time for batching in milliseconds
    #[arg(long)]
    pub max_wait_time_ms: Option<u64>,

    /// Maximum batch size, check your model's limits - e.g., all-MiniLM-L6-v2 supports up to 32)
    #[arg(long)]
    pub max_batch_size: Option<usize>,

    /// How often it can apply pending requests age check
    #[arg(long)]
    pub batch_check_interval_ms: Option<u64>,

    /// Whether to include batching info in response. Helpful in development. Used in tests
    #[arg(long)]
    pub include_batch_info: Option<bool>,

    /// Inference service Full URL
    #[arg(long)]
    pub inference_url: Option<String>,

    /// Inference service timeout
    #[arg(long)]
    pub inference_timeout_secs: Option<u64>,

    /// Maximum inputs per inference service call
    #[arg(long)]
    pub max_inference_inputs: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    pub port: u16,
    pub max_wait_time_ms: u64,
    pub max_batch_size: usize,
    pub batch_check_interval_ms: u64,
    pub include_batch_info: bool,
    pub inference_url: String,
    pub inference_timeout_secs: u64,
    pub max_inference_inputs: usize,
    pub log_level: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            port: 3000,
            max_wait_time_ms: 500,
            max_batch_size: 8,
            batch_check_interval_ms: 10, // in general, 100 ms is good enough
            include_batch_info: false,
            inference_url: "http://127.0.0.1:8080/embed".to_string(),
            inference_timeout_secs: 30,
            max_inference_inputs: 32,
            log_level: "info".to_string(),
        }
    }
}

impl AppConfig {
    /// Build config from CLI args and defaults
    pub fn build(args: Option<Args>) -> Self {
        let mut config = Self::default();
        if let Some(args) = args {
            if let Some(port) = args.port {
                config.port = port;
            }
            if let Some(max_wait_time_ms) = args.max_wait_time_ms {
                config.max_wait_time_ms = max_wait_time_ms;
            }
            //  `--model-id sentence-transformers/all-MiniLM-L6-v2` handles max 32 inputs
            // max 32 check is not applied here, since different models could have different configs
            if let Some(max_batch_size) = args.max_batch_size {
                config.max_batch_size = max_batch_size;
            }
            if let Some(batch_check_interval_ms) = args.batch_check_interval_ms {
                config.batch_check_interval_ms = batch_check_interval_ms;
            }

            if let Some(include_batch_info) = args.include_batch_info {
                config.include_batch_info = include_batch_info;
            }

            if let Some(inference_url) = args.inference_url {
                config.inference_url = inference_url;
            }
            if let Some(inference_timeout_secs) = args.inference_timeout_secs {
                config.inference_timeout_secs = inference_timeout_secs;
            }
            if let Some(max_inference_inputs) = args.max_inference_inputs {
                config.max_inference_inputs = max_inference_inputs;
            }
        }
        config
    }

    pub fn max_wait_time_duration(&self) -> Duration {
        Duration::from_millis(self.max_wait_time_ms)
    }

    /// Initialize logging with env_logger (simpler approach)
    pub fn init_logging(&self) -> String {
        env_logger::Builder::from_env(
            env_logger::Env::default().default_filter_or(&self.log_level),
        )
        .init();
        std::env::var("RUST_LOG").unwrap_or_else(|_| self.log_level.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::build(None);
        assert_eq!(config.port, 3000);
        assert_eq!(config.max_wait_time_ms, 500);
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.batch_check_interval_ms, 10);
        assert_eq!(config.inference_url, "http://127.0.0.1:8080/embed");
        assert_eq!(config.inference_timeout_secs, 30);
        assert_eq!(config.log_level, "info");
    }

    #[test]
    fn test_config_from_args() {
        let args = Args {
            port: Some(4000),
            max_wait_time_ms: Some(200),
            max_batch_size: Some(16),
            batch_check_interval_ms: Some(10),
            include_batch_info: None,
            inference_url: Some("http://custom:9090/embed".to_string()),
            inference_timeout_secs: Some(60),
            max_inference_inputs: None,
        };

        let config = AppConfig::build(Some(args));
        assert_eq!(config.port, 4000);
        assert_eq!(config.max_wait_time_ms, 200);
        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.batch_check_interval_ms, 10);
        assert_eq!(config.inference_url, "http://custom:9090/embed");
        assert_eq!(config.inference_timeout_secs, 60);
    }

    #[test]
    fn test_partial_args_from_args() {
        let args = Args {
            port: Some(5000),
            max_wait_time_ms: None,
            max_batch_size: Some(25),
            batch_check_interval_ms: None,
            include_batch_info: None,
            inference_url: None,
            inference_timeout_secs: None,
            max_inference_inputs: None,
        };

        let config = AppConfig::build(Some(args));
        let defaults = AppConfig::default();
        assert_eq!(config.port, 5000);
        assert_eq!(config.max_wait_time_ms, defaults.max_wait_time_ms);
        assert_eq!(config.max_batch_size, 25);
        assert_eq!(
            config.batch_check_interval_ms,
            defaults.batch_check_interval_ms
        );
        assert_eq!(config.inference_url, defaults.inference_url);
        assert_eq!(
            config.inference_timeout_secs,
            defaults.inference_timeout_secs
        );
    }
}

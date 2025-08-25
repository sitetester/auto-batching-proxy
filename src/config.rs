use clap::Parser;
use rocket::log::LogLevel;
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

    /// Maximum batch size, check your model's limits - e.g., all-MiniLM-L6-v2 supports up to 32
    #[arg(long)]
    pub max_batch_size: Option<usize>,

    /// How often it can apply pending requests age check
    #[arg(long)]
    pub batch_check_interval_ms: Option<u64>,

    /// Whether to include batching info in response. Helpful in development. Used in tests.
    /// Not applicable in Production setup
    #[arg(long)]
    pub include_batch_info: Option<bool>,

    /// Inference service full URL
    #[arg(long)]
    pub inference_url: Option<String>,

    /// Inference service timeout
    #[arg(long)]
    pub inference_timeout_secs: Option<u64>,

    /// Maximum inputs per inference service call
    #[arg(long)]
    pub max_inference_inputs: Option<usize>,

    /// Maximum inputs per inference service call
    #[arg(long)]
    pub log_level: Option<LogLevel>,
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
    pub fn build(args: Option<Args>) -> Result<Self, String> {
        let mut config = Self::default();
        if let Some(args) = args {
            if let Some(port) = args.port {
                config.port = port;
            }
            if let Some(max_wait_time_ms) = args.max_wait_time_ms {
                if max_wait_time_ms == 0 {
                    return Err("max_wait_time_ms must be > 0".to_string());
                }
                config.max_wait_time_ms = max_wait_time_ms;
            }
            //  `--model-id sentence-transformers/all-MiniLM-L6-v2` handles max 32 inputs
            // max 32 check is not applied here, since different models could have different configs
            if let Some(max_batch_size) = args.max_batch_size {
                if max_batch_size == 0 {
                    return Err("max_batch_size must be > 0".to_string());
                }
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
            if let Some(log_level) = args.log_level {
                config.log_level = log_level.to_string().to_lowercase();
            }
        }
        Ok(config)
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
    fn test_build_from_default() {
        let config = AppConfig::build(None);
        assert!(config.is_ok());
        let config = config.unwrap();

        let defaults = AppConfig::default();
        assert_eq!(config.port, defaults.port);
        assert_eq!(config.max_wait_time_ms, defaults.max_wait_time_ms);
        assert_eq!(config.max_batch_size, defaults.max_batch_size);
        assert_eq!(
            config.batch_check_interval_ms,
            defaults.batch_check_interval_ms
        );
        assert_eq!(config.inference_url, defaults.inference_url);
        assert_eq!(
            config.inference_timeout_secs,
            defaults.inference_timeout_secs
        );
        assert_eq!(config.log_level, defaults.log_level);
    }

    #[test]
    fn test_build_from_args() {
        let args = Args {
            port: Some(6000),
            max_wait_time_ms: Some(200),
            max_batch_size: Some(16),
            batch_check_interval_ms: Some(50),
            include_batch_info: Some(false),
            inference_url: Some("http://custom:9090/embed".to_string()),
            inference_timeout_secs: Some(60),
            max_inference_inputs: Some(16),
            log_level: Some(LogLevel::Debug),
        };

        let config = AppConfig::build(Some(args));
        assert!(config.is_ok());
        let config = config.unwrap();

        assert_eq!(config.port, 6000);
        assert_eq!(config.max_wait_time_ms, 200);
        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.batch_check_interval_ms, 50);
        assert_eq!(config.include_batch_info, false);
        assert_eq!(config.inference_url, "http://custom:9090/embed");
        assert_eq!(config.inference_timeout_secs, 60);
        assert_eq!(config.max_inference_inputs, 16);
        assert_eq!(config.log_level, "debug".to_string());
    }

    fn get_empty_args() -> Args {
        Args {
            port: None,
            max_wait_time_ms: None,
            max_batch_size: None,
            batch_check_interval_ms: None,
            include_batch_info: None,
            inference_url: None,
            inference_timeout_secs: None,
            max_inference_inputs: None,
            log_level: None,
        }
    }

    #[test]
    fn test_build_from_partial_args() {
        let partial_args = Args {
            port: Some(5000),
            max_batch_size: Some(25),
            ..get_empty_args()
        };

        let config = AppConfig::build(Some(partial_args));
        assert!(config.is_ok());
        let config = config.unwrap();

        let defaults = AppConfig::default();
        assert_eq!(config.port, 5000);
        assert_eq!(config.max_wait_time_ms, defaults.max_wait_time_ms);
        assert_eq!(config.max_batch_size, 25);
        assert_eq!(
            config.batch_check_interval_ms,
            defaults.batch_check_interval_ms
        );
        assert_eq!(config.include_batch_info, defaults.include_batch_info);
        assert_eq!(config.inference_url, defaults.inference_url);
        assert_eq!(
            config.inference_timeout_secs,
            defaults.inference_timeout_secs
        );
    }

    #[test]
    fn test_build_fails_when_max_batch_size_is_0() {
        let invalid_args = Args {
            max_batch_size: Some(0),
            ..get_empty_args()
        };

        let config = AppConfig::build(Some(invalid_args));
        assert!(config.is_err());
    }

    #[test]
    fn test_build_fails_when_max_wait_time_ms_is_0() {
        let invalid_args = Args {
            max_wait_time_ms: Some(0),
            ..get_empty_args()
        };

        let config = AppConfig::build(Some(invalid_args));
        assert!(config.is_err());
    }
}

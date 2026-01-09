mod config;
mod sink;
#[cfg(feature = "codecs-parquet")]
mod super_batch_sink;

mod integration_tests;

pub use config::S3SinkConfig;

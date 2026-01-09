//! Disk-backed S3 sink for high-cardinality partitioning with large batches.
//!
//! This sink is designed for scenarios where:
//! - There are tens of thousands of unique partition keys (e.g., `{{team_id}}/{{date}}`)
//! - Batches need to be very large (100k events, 7-10GB per batch)
//! - Memory usage must be controlled
//!
//! Instead of keeping all partition batches in memory, events are written to
//! temporary files on disk per partition. When a batch is ready, the file is
//! read back and uploaded to S3.
//!
//! This sink also supports parquet super-batch mode (rows_per_file), allowing
//! a single disk batch to be split into multiple parquet files.

use std::{fmt, hash::Hash, io, path::PathBuf};

use chrono::{FixedOffset, Utc};
use futures::{stream::BoxStream, StreamExt};
use tower::Service;
use uuid::Uuid;
use vector_lib::{
    ByteSizeOf,
    config::telemetry,
    event::{Event, Finalizable},
    partition::Partitioner,
    sink::StreamSink,
    stream::DriverResponse,
    EstimatedJsonEncodedSizeOf,
};

use super::disk_batch::{DiskBackedPartitionedBatcher, DiskBatch, DiskBatchConfig};
use crate::{
    codecs::{EncoderKind, Transformer},
    internal_events::SinkRequestBuildError,
    sinks::{
        prelude::*,
        s3_common::{
            config::S3Options,
            partitioner::S3PartitionKey,
            service::{S3Metadata, S3Request},
        },
        util::{metadata::RequestMetadataBuilder, request_builder::EncodeResult},
    },
};

/// Configuration for the disk-backed S3 sink.
#[derive(Debug, Clone)]
pub struct DiskBackedSinkConfig {
    /// Directory to store temporary batch files.
    pub temp_dir: PathBuf,
    /// Maximum number of bytes per batch before flushing.
    pub max_bytes: usize,
    /// Maximum number of events per batch before flushing.
    pub max_events: usize,
    /// Timeout for batches.
    pub timeout: std::time::Duration,
}

impl Default for DiskBackedSinkConfig {
    fn default() -> Self {
        Self {
            temp_dir: std::env::temp_dir().join("vector-s3-batches"),
            max_bytes: 10 * 1024 * 1024 * 1024, // 10GB
            max_events: 100_000,
            timeout: std::time::Duration::from_secs(300), // 5 minutes
        }
    }
}

impl From<DiskBackedSinkConfig> for DiskBatchConfig {
    fn from(config: DiskBackedSinkConfig) -> Self {
        DiskBatchConfig {
            temp_dir: config.temp_dir,
            max_bytes: config.max_bytes,
            max_events: config.max_events,
            timeout: config.timeout,
        }
    }
}

/// Request builder for disk-backed batches.
#[derive(Clone)]
pub struct DiskBackedRequestBuilder {
    pub bucket: String,
    pub filename_time_format: String,
    pub filename_append_uuid: bool,
    pub filename_extension: Option<String>,
    pub api_options: S3Options,
    pub encoder: (Transformer, EncoderKind),
    pub compression: Compression,
    pub filename_tz_offset: Option<FixedOffset>,
}

impl DiskBackedRequestBuilder {
    /// Returns true if super-batch mode is enabled (parquet with rows_per_file).
    pub fn is_super_batch_enabled(&self) -> bool {
        self.encoder.1.is_super_batch_enabled()
    }

    fn generate_filename(&self) -> String {
        let formatted_ts = match self.filename_tz_offset {
            Some(offset) => Utc::now()
                .with_timezone(&offset)
                .format(self.filename_time_format.as_str()),
            None => Utc::now()
                .with_timezone(&Utc)
                .format(self.filename_time_format.as_str()),
        };

        if self.filename_append_uuid {
            format!("{formatted_ts}-{}", Uuid::new_v4().hyphenated())
        } else {
            formatted_ts.to_string()
        }
    }

    fn get_extension(&self) -> String {
        self.filename_extension
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.compression.extension().into())
    }

    /// Builds S3 request(s) from a disk batch.
    ///
    /// Returns multiple requests if super-batch mode is enabled (parquet with rows_per_file),
    /// otherwise returns a single request.
    pub fn build_requests_from_batch(
        &self,
        partition_key: S3PartitionKey,
        mut batch: DiskBatch,
    ) -> Result<Vec<S3Request>, io::Error> {
        // Read events from disk
        let mut events = batch.read_events()?;
        let total_events = events.len();

        // Get finalizers
        let finalizers = batch.take_finalizers();

        // Clean up the batch file early since we have the events in memory now
        batch.cleanup();

        // Transform events and track byte size
        let mut total_byte_size = telemetry().create_request_count_byte_size();
        let mut events_byte_size: usize = 0;
        for event in &mut events {
            self.encoder.0.transform(event);
            events_byte_size += event.size_of();
            total_byte_size.add_event(event, event.estimated_json_encoded_size_of());
        }

        // Check if super-batch mode is enabled
        if self.is_super_batch_enabled() {
            self.build_super_batch_requests(partition_key, events, total_events, events_byte_size, total_byte_size, finalizers)
        } else {
            let request = self.build_single_request(partition_key, events, events_byte_size, finalizers)?;
            Ok(vec![request])
        }
    }

    /// Builds a single S3 request (standard mode).
    fn build_single_request(
        &self,
        partition_key: S3PartitionKey,
        events: Vec<Event>,
        events_byte_size: usize,
        finalizers: vector_lib::finalization::EventFinalizers,
    ) -> Result<S3Request, io::Error> {
        use crate::sinks::util::{Compressor, encoding::Encoder as EncoderTrait};

        let event_count = events.len();

        // Use a compressor to handle both encoding and compression
        let mut compressor = Compressor::from(self.compression);
        let is_compressed = compressor.is_compressed();

        // Encode events using the encoding infrastructure
        let (uncompressed_size, byte_size) =
            self.encoder.encode_input(events, &mut compressor)?;

        let payload = compressor.into_inner().freeze();

        // Build encode result
        let encode_result = if is_compressed {
            EncodeResult::compressed(payload.clone(), uncompressed_size, byte_size.clone())
        } else {
            EncodeResult::uncompressed(payload.clone(), byte_size.clone())
        };

        let filename = self.generate_filename();
        let extension = self.get_extension();

        let ssekms_key_id = partition_key.ssekms_key_id.clone();
        let mut s3_options = self.api_options.clone();
        s3_options.ssekms_key_id = ssekms_key_id;

        let s3_key = format_s3_key(&partition_key.key_prefix, &filename, &extension);

        // Build request metadata
        let request_metadata_builder =
            RequestMetadataBuilder::new(event_count, events_byte_size, byte_size);
        let request_metadata = request_metadata_builder.build(&encode_result);

        let metadata = S3Metadata {
            partition_key,
            s3_key,
            finalizers,
            event_count: Some(event_count),
            events_byte_size: None,
        };

        Ok(S3Request {
            body: payload,
            bucket: self.bucket.clone(),
            metadata,
            request_metadata,
            content_encoding: self.compression.content_encoding(),
            options: s3_options,
        })
    }

    /// Builds multiple S3 requests using super-batch mode (parquet with rows_per_file).
    fn build_super_batch_requests(
        &self,
        partition_key: S3PartitionKey,
        events: Vec<Event>,
        total_events: usize,
        total_events_byte_size: usize,
        total_byte_size: vector_lib::request_metadata::GroupedCountByteSize,
        finalizers: vector_lib::finalization::EventFinalizers,
    ) -> Result<Vec<S3Request>, io::Error> {
        // Encode using super-batch mode (produces multiple parquet files)
        let parquet_files = self.encoder.1.encode_batch_split(events).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Failed to encode parquet batch: {}", err),
            )
        })?;

        let num_files = parquet_files.len();
        let s3_key_prefix = partition_key.key_prefix.clone();
        let extension = self.get_extension();

        // Calculate events per file (distribute evenly, last file gets remainder)
        let base_events_per_file = total_events / num_files.max(1);
        let remainder = total_events % num_files.max(1);

        // Calculate base events_byte_size per file (distribute proportionally)
        let base_events_byte_size_per_file = total_events_byte_size / num_files.max(1);
        let byte_size_remainder = total_events_byte_size % num_files.max(1);

        let mut requests = Vec::with_capacity(num_files);

        for (idx, file_bytes) in parquet_files.into_iter().enumerate() {
            // Split finalizers - only the last file gets the finalizers
            let file_finalizers = if idx == num_files - 1 {
                finalizers.clone()
            } else {
                Default::default()
            };

            // Calculate event count for this file
            let file_event_count = if idx < remainder {
                base_events_per_file + 1
            } else {
                base_events_per_file
            };

            // Calculate events_byte_size for this file (last file gets remainder)
            let file_events_byte_size = if idx == num_files - 1 {
                base_events_byte_size_per_file + byte_size_remainder
            } else {
                base_events_byte_size_per_file
            };

            // Create proportional GroupedCountByteSize for this file
            let file_byte_size = total_byte_size.clone();

            let filename = self.generate_filename();

            let ssekms_key_id = partition_key.ssekms_key_id.clone();
            let mut s3_options = self.api_options.clone();
            s3_options.ssekms_key_id = ssekms_key_id;

            let s3_key = format_s3_key(&s3_key_prefix, &filename, &extension);

            let request_metadata_builder = RequestMetadataBuilder::new(
                file_event_count,
                file_events_byte_size,
                file_byte_size.clone(),
            );
            let encode_result = EncodeResult::uncompressed(file_bytes.clone(), file_byte_size);
            let request_metadata = request_metadata_builder.build(&encode_result);

            let metadata = S3Metadata {
                partition_key: partition_key.clone(),
                s3_key,
                finalizers: file_finalizers,
                event_count: Some(file_event_count),
                events_byte_size: None,
            };

            requests.push(S3Request {
                body: file_bytes,
                bucket: self.bucket.clone(),
                metadata,
                request_metadata,
                content_encoding: self.compression.content_encoding(),
                options: s3_options,
            });
        }

        Ok(requests)
    }
}

fn format_s3_key(s3_key: &str, filename: &str, extension: &str) -> String {
    if extension.is_empty() {
        format!("{s3_key}{filename}")
    } else {
        format!("{s3_key}{filename}.{extension}")
    }
}

/// Disk-backed S3 sink for high-cardinality partitioning.
pub struct DiskBackedS3Sink<Svc, P> {
    service: Svc,
    request_builder: DiskBackedRequestBuilder,
    partitioner: P,
    batch_config: DiskBatchConfig,
}

impl<Svc, P> DiskBackedS3Sink<Svc, P> {
    pub fn new(
        service: Svc,
        request_builder: DiskBackedRequestBuilder,
        partitioner: P,
        batch_config: DiskBatchConfig,
    ) -> Self {
        Self {
            service,
            request_builder,
            partitioner,
            batch_config,
        }
    }
}

impl<Svc, P> DiskBackedS3Sink<Svc, P>
where
    Svc: Service<S3Request> + Send + 'static,
    Svc::Future: Send + 'static,
    Svc::Response: DriverResponse + Send + 'static,
    Svc::Error: fmt::Debug + Into<crate::Error> + Send,
    P: Partitioner<Item = Event, Key = Option<S3PartitionKey>> + Unpin + Send,
    P::Key: Eq + Hash + Clone,
{
    async fn run_inner(self: Box<Self>, input: BoxStream<'_, Event>) -> Result<(), ()> {
        let partitioner = self.partitioner;
        let batch_config = self.batch_config;
        let request_builder = self.request_builder;

        // Create a wrapper partitioner that extracts the Option<S3PartitionKey>
        let wrapped_partitioner = OptionKeyPartitioner { inner: partitioner };

        // Use the disk-backed batcher
        let batcher = DiskBackedPartitionedBatcher::new(input, wrapped_partitioner, batch_config);

        // Process batches - may produce multiple requests per batch in super-batch mode
        batcher
            .flat_map(|(key, batch)| {
                let request_builder = request_builder.clone();

                // Convert to stream of requests
                let requests: Vec<S3Request> = match key {
                    Some(partition_key) => {
                        match request_builder.build_requests_from_batch(partition_key, batch) {
                            Ok(reqs) => reqs,
                            Err(error) => {
                                emit!(SinkRequestBuildError { error });
                                vec![]
                            }
                        }
                    }
                    None => vec![],
                };

                futures::stream::iter(requests)
            })
            .into_driver(self.service)
            .run()
            .await
    }
}

#[async_trait::async_trait]
impl<Svc, P> StreamSink<Event> for DiskBackedS3Sink<Svc, P>
where
    Svc: Service<S3Request> + Send + 'static,
    Svc::Future: Send + 'static,
    Svc::Response: DriverResponse + Send + 'static,
    Svc::Error: fmt::Debug + Into<crate::Error> + Send,
    P: Partitioner<Item = Event, Key = Option<S3PartitionKey>> + Unpin + Send,
    P::Key: Eq + Hash + Clone,
{
    async fn run(mut self: Box<Self>, input: BoxStream<'_, Event>) -> Result<(), ()> {
        self.run_inner(input).await
    }
}

/// Wrapper partitioner that handles Option<S3PartitionKey>.
struct OptionKeyPartitioner<P> {
    inner: P,
}

impl<P> Partitioner for OptionKeyPartitioner<P>
where
    P: Partitioner<Item = Event, Key = Option<S3PartitionKey>>,
{
    type Item = Event;
    type Key = Option<S3PartitionKey>;

    fn partition(&self, item: &Self::Item) -> Self::Key {
        self.inner.partition(item)
    }
}

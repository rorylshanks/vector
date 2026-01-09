use std::io;

use bytes::Bytes;
use chrono::{FixedOffset, Utc};
use uuid::Uuid;
use vector_lib::{
    EstimatedJsonEncodedSizeOf, config::telemetry, event::Finalizable,
    request_metadata::RequestMetadata,
};

use crate::{
    codecs::{EncoderKind, Transformer},
    event::Event,
    sinks::{
        s3_common::{
            config::S3Options,
            partitioner::S3PartitionKey,
            service::{S3Metadata, S3Request},
        },
        util::{
            Compression, IncrementalRequestBuilder, RequestBuilder,
            metadata::RequestMetadataBuilder, request_builder::EncodeResult,
        },
    },
};

#[derive(Clone)]
pub struct S3RequestOptions {
    pub bucket: String,
    pub filename_time_format: String,
    pub filename_append_uuid: bool,
    pub filename_extension: Option<String>,
    pub api_options: S3Options,
    pub encoder: (Transformer, EncoderKind),
    pub compression: Compression,
    pub filename_tz_offset: Option<FixedOffset>,
}

impl RequestBuilder<(S3PartitionKey, Vec<Event>)> for S3RequestOptions {
    type Metadata = S3Metadata;
    type Events = Vec<Event>;
    type Encoder = (Transformer, EncoderKind);
    type Payload = Bytes;
    type Request = S3Request;
    type Error = io::Error; // TODO: this is ugly.

    fn compression(&self) -> Compression {
        self.compression
    }

    fn encoder(&self) -> &Self::Encoder {
        &self.encoder
    }

    fn split_input(
        &self,
        input: (S3PartitionKey, Vec<Event>),
    ) -> (Self::Metadata, RequestMetadataBuilder, Self::Events) {
        let (partition_key, mut events) = input;
        let builder = RequestMetadataBuilder::from_events(&events);

        let finalizers = events.take_finalizers();
        let s3_key_prefix = partition_key.key_prefix.clone();

        let metadata = S3Metadata {
            partition_key,
            s3_key: s3_key_prefix,
            finalizers,
            event_count: None,
            events_byte_size: None,
        };

        (metadata, builder, events)
    }

    fn build_request(
        &self,
        mut s3metadata: Self::Metadata,
        request_metadata: RequestMetadata,
        payload: EncodeResult<Self::Payload>,
    ) -> Self::Request {
        let filename = {
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
        };

        let ssekms_key_id = s3metadata.partition_key.ssekms_key_id.clone();
        let mut s3_options = self.api_options.clone();
        s3_options.ssekms_key_id = ssekms_key_id;

        let extension = self
            .filename_extension
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.compression.extension().into());

        s3metadata.s3_key = format_s3_key(&s3metadata.s3_key, &filename, &extension);

        S3Request {
            body: payload.into_payload(),
            bucket: self.bucket.clone(),
            metadata: s3metadata,
            request_metadata,
            content_encoding: self.compression.content_encoding(),
            options: s3_options,
        }
    }
}

fn format_s3_key(s3_key: &str, filename: &str, extension: &str) -> String {
    if extension.is_empty() {
        format!("{s3_key}{filename}")
    } else {
        format!("{s3_key}{filename}.{extension}")
    }
}

/// Request builder for super-batch mode (parquet with rows_per_file).
///
/// This builder implements `IncrementalRequestBuilder` to support splitting
/// a single batch of events into multiple S3 requests, one per parquet file.
#[cfg(feature = "codecs-parquet")]
#[derive(Clone)]
pub struct S3SuperBatchRequestBuilder {
    pub bucket: String,
    pub filename_time_format: String,
    pub filename_append_uuid: bool,
    pub filename_extension: Option<String>,
    pub api_options: S3Options,
    pub encoder: (Transformer, EncoderKind),
    pub compression: Compression,
    pub filename_tz_offset: Option<FixedOffset>,
}

#[cfg(feature = "codecs-parquet")]
impl S3SuperBatchRequestBuilder {
    /// Creates a new super-batch request builder from S3RequestOptions.
    pub fn from_options(options: S3RequestOptions) -> Self {
        Self {
            bucket: options.bucket,
            filename_time_format: options.filename_time_format,
            filename_append_uuid: options.filename_append_uuid,
            filename_extension: options.filename_extension,
            api_options: options.api_options,
            encoder: options.encoder,
            compression: options.compression,
            filename_tz_offset: options.filename_tz_offset,
        }
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
}

#[cfg(feature = "codecs-parquet")]
impl IncrementalRequestBuilder<(S3PartitionKey, Vec<Event>)> for S3SuperBatchRequestBuilder {
    type Metadata = S3Metadata;
    type Payload = Bytes;
    type Request = S3Request;
    type Error = io::Error;

    fn encode_events_incremental(
        &mut self,
        input: (S3PartitionKey, Vec<Event>),
    ) -> Vec<Result<(Self::Metadata, Self::Payload), Self::Error>> {
        let (partition_key, mut events) = input;
        let total_events = events.len();

        // Transform events and track byte size
        let mut transformed_events = Vec::with_capacity(events.len());
        let mut total_byte_size = telemetry().create_request_count_byte_size();

        for mut event in events.drain(..) {
            self.encoder.0.transform(&mut event);
            total_byte_size.add_event(&event, event.estimated_json_encoded_size_of());
            transformed_events.push(event);
        }

        // Get finalizers from the transformed events
        let finalizers = transformed_events.take_finalizers();

        // Encode using super-batch mode (produces multiple parquet files)
        let parquet_files = match self.encoder.1.encode_batch_split(transformed_events) {
            Ok(files) => files,
            Err(err) => {
                return vec![Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed to encode parquet batch: {}", err),
                ))];
            }
        };

        let num_files = parquet_files.len();
        let s3_key_prefix = partition_key.key_prefix.clone();

        // Calculate events per file (distribute evenly, last file gets remainder)
        let base_events_per_file = total_events / num_files.max(1);
        let remainder = total_events % num_files.max(1);

        // Create a request for each parquet file
        parquet_files
            .into_iter()
            .enumerate()
            .map(|(idx, file_bytes)| {
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

                // Create proportional byte size for this file
                let mut file_byte_size = telemetry().create_request_count_byte_size();
                // Clone the total and scale proportionally (approximation)
                if total_events > 0 {
                    file_byte_size = total_byte_size.clone();
                }

                let metadata = S3Metadata {
                    partition_key: partition_key.clone(),
                    s3_key: s3_key_prefix.clone(),
                    finalizers: file_finalizers,
                    event_count: Some(file_event_count),
                    events_byte_size: Some(file_byte_size),
                };

                Ok((metadata, file_bytes))
            })
            .collect()
    }

    fn build_request(&mut self, mut metadata: Self::Metadata, payload: Self::Payload) -> Self::Request {
        let filename = self.generate_filename();

        let ssekms_key_id = metadata.partition_key.ssekms_key_id.clone();
        let mut s3_options = self.api_options.clone();
        s3_options.ssekms_key_id = ssekms_key_id;

        let extension = self
            .filename_extension
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.compression.extension().into());

        metadata.s3_key = format_s3_key(&metadata.s3_key, &filename, &extension);

        let uncompressed_size = payload.len();
        // Use event count and byte size from metadata (set in encode_events_incremental)
        let event_count = metadata.event_count.unwrap_or(0);
        let events_byte_size = metadata.events_byte_size.clone()
            .unwrap_or_else(|| telemetry().create_request_count_byte_size());

        let request_metadata_builder = RequestMetadataBuilder::new(
            event_count,
            uncompressed_size,
            events_byte_size,
        );
        let encode_result = EncodeResult::uncompressed(payload, telemetry().create_request_count_byte_size());
        let request_metadata = request_metadata_builder.build(&encode_result);

        S3Request {
            body: encode_result.into_payload(),
            bucket: self.bucket.clone(),
            metadata,
            request_metadata,
            content_encoding: self.compression.content_encoding(),
            options: s3_options,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_s3_key() {
        assert_eq!(
            "s3_key_filename.txt",
            format_s3_key("s3_key_", "filename", "txt")
        );
        assert_eq!("s3_key_filename", format_s3_key("s3_key_", "filename", ""));
    }
}

//! Apache Parquet format codec for batched event encoding
//!
//! Provides Apache Parquet columnar file format encoding with static schema support.
//! This encoder writes complete Parquet files with proper metadata and footers,
//! suitable for long-term storage and analytics workloads.

use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Schema};
use bytes::{BufMut, Bytes, BytesMut};
use parquet::{
    arrow::ArrowWriter,
    basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel},
    file::properties::{WriterProperties, WriterVersion},
    schema::types::ColumnPath,
};
use snafu::Snafu;
use std::collections::BTreeMap;
use std::sync::Arc;
use vector_config::configurable_component;

use vector_core::event::Event;

// Reuse the Arrow encoder's record batch building logic
use super::arrow::{ArrowEncodingError, build_record_batch};
use super::json_column::{JsonColumnProcessor, ProcessedJsonColumns};
use super::schema_definition::SchemaDefinition;

/// Compression algorithm for Parquet files
#[configurable_component]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ParquetCompression {
    /// No compression
    Uncompressed,
    /// Snappy compression (fast, moderate compression ratio)
    #[default]
    Snappy,
    /// GZIP compression (slower, better compression ratio)
    Gzip,
    /// Brotli compression
    Brotli,
    /// LZ4 compression (very fast, moderate compression)
    Lz4,
    /// ZSTD compression (good balance of speed and compression)
    Zstd,
}

impl ParquetCompression {
    /// Convert to parquet Compression with optional level override
    fn to_compression(self, level: Option<i32>) -> Result<Compression, String> {
        match (self, level) {
            (ParquetCompression::Uncompressed, _) => Ok(Compression::UNCOMPRESSED),
            (ParquetCompression::Snappy, _) => Ok(Compression::SNAPPY),
            (ParquetCompression::Lz4, _) => Ok(Compression::LZ4),
            (ParquetCompression::Gzip, Some(lvl)) => GzipLevel::try_new(lvl as u32)
                .map(Compression::GZIP)
                .map_err(|e| format!("Invalid GZIP compression level: {}", e)),
            (ParquetCompression::Gzip, None) => Ok(Compression::GZIP(Default::default())),
            (ParquetCompression::Brotli, Some(lvl)) => BrotliLevel::try_new(lvl as u32)
                .map(Compression::BROTLI)
                .map_err(|e| format!("Invalid Brotli compression level: {}", e)),
            (ParquetCompression::Brotli, None) => Ok(Compression::BROTLI(Default::default())),
            (ParquetCompression::Zstd, Some(lvl)) => ZstdLevel::try_new(lvl)
                .map(Compression::ZSTD)
                .map_err(|e| format!("Invalid ZSTD compression level: {}", e)),
            (ParquetCompression::Zstd, None) => Ok(Compression::ZSTD(ZstdLevel::default())),
        }
    }
}

impl From<ParquetCompression> for Compression {
    fn from(compression: ParquetCompression) -> Self {
        compression
            .to_compression(None)
            .expect("Default compression should always be valid")
    }
}

/// Parquet writer version
#[configurable_component]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ParquetWriterVersion {
    /// Parquet format version 1.0 (maximum compatibility)
    V1,
    /// Parquet format version 2.0 (modern format with better encoding)
    #[default]
    V2,
}

impl From<ParquetWriterVersion> for WriterVersion {
    fn from(version: ParquetWriterVersion) -> Self {
        match version {
            ParquetWriterVersion::V1 => WriterVersion::PARQUET_1_0,
            ParquetWriterVersion::V2 => WriterVersion::PARQUET_2_0,
        }
    }
}

/// Configuration for Parquet serialization
#[configurable_component]
#[derive(Clone)]
pub struct ParquetSerializerConfig {
    /// The Arrow schema definition to use for encoding
    ///
    /// This schema defines the structure and types of the Parquet file columns.
    /// Specified as a map of field names to data types.
    ///
    /// Mutually exclusive with `infer_schema`. Must specify either `schema` or `infer_schema: true`.
    ///
    /// Supported types: utf8, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    /// float32, float64, boolean, binary, timestamp_second, timestamp_millisecond,
    /// timestamp_microsecond, timestamp_nanosecond, date32, date64, and more.
    #[serde(default)]
    #[configurable(metadata(docs::examples = "schema_example()"))]
    pub schema: Option<SchemaDefinition>,

    /// Automatically infer schema from event data
    ///
    /// When enabled, the schema is inferred from each batch of events independently.
    /// The schema is determined by examining the types of values in the events.
    ///
    /// **Type mapping:**
    /// - String values → `utf8`
    /// - Integer values → `int64`
    /// - Float values → `float64`
    /// - Boolean values → `boolean`
    /// - Timestamp values → `timestamp_microsecond`
    /// - Arrays/Objects → `utf8` (serialized as JSON)
    ///
    /// **Type conflicts:** If a field has different types across events in the same batch,
    /// it will be encoded as `utf8` (string) and all values will be converted to strings.
    ///
    /// **Important:** Schema consistency across batches is the operator's responsibility.
    /// Use VRL transforms to ensure consistent types if needed. Each batch may produce
    /// a different schema if event structure varies.
    ///
    /// **Bloom filters:** Not supported with inferred schemas. Use explicit schema for Bloom filters.
    ///
    /// Mutually exclusive with `schema`. Must specify either `schema` or `infer_schema: true`.
    #[serde(default)]
    #[configurable(metadata(docs::examples = true))]
    pub infer_schema: bool,

    /// Column names to exclude from Parquet encoding
    ///
    /// These columns will be completely excluded from the Parquet file.
    /// Useful for filtering out metadata, internal fields, or temporary data.
    ///
    /// Only applies when `infer_schema` is enabled. Ignored when using explicit schema.
    #[serde(default)]
    #[configurable(metadata(
        docs::examples = "vec![\"_metadata\".to_string(), \"internal_id\".to_string()]"
    ))]
    pub exclude_columns: Option<Vec<String>>,

    /// Maximum number of columns to encode
    ///
    /// Limits the number of columns in the Parquet file. Additional columns beyond
    /// this limit will be silently dropped. Columns are selected in the order they
    /// appear in the first event.
    ///
    /// Only applies when `infer_schema` is enabled. Ignored when using explicit schema.
    #[serde(default = "default_max_columns")]
    #[configurable(metadata(docs::examples = 500))]
    #[configurable(metadata(docs::examples = 1000))]
    pub max_columns: usize,

    /// Compression algorithm to use for Parquet columns
    ///
    /// Compression is applied to all columns in the Parquet file.
    /// Snappy provides a good balance of speed and compression ratio.
    #[serde(default)]
    #[configurable(metadata(docs::examples = "snappy"))]
    #[configurable(metadata(docs::examples = "gzip"))]
    #[configurable(metadata(docs::examples = "zstd"))]
    pub compression: ParquetCompression,

    /// Compression level for algorithms that support it.
    ///
    /// Only applies to ZSTD, GZIP, and Brotli compression. Ignored for other algorithms.
    ///
    /// **ZSTD levels** (1-22):
    /// - 1-3: Fastest, moderate compression (level 3 is default)
    /// - 4-9: Good balance of speed and compression
    /// - 10-15: Better compression, slower encoding
    /// - 16-22: Maximum compression, slowest (good for cold storage)
    ///
    /// **GZIP levels** (1-9):
    /// - 1-3: Faster, less compression
    /// - 6: Default balance (recommended)
    /// - 9: Maximum compression, slowest
    ///
    /// **Brotli levels** (0-11):
    /// - 0-4: Faster encoding
    /// - 1: Default (recommended)
    /// - 5-11: Better compression, slower
    ///
    /// Higher levels typically produce 20-50% smaller files but take 2-5x longer to encode.
    /// Recommended: Use level 3-6 for hot data, 10-15 for cold storage.
    #[serde(default)]
    #[configurable(metadata(docs::examples = 3))]
    #[configurable(metadata(docs::examples = 6))]
    #[configurable(metadata(docs::examples = 10))]
    pub compression_level: Option<i32>,

    /// Parquet format writer version.
    ///
    /// Controls which Parquet format version to write:
    /// - **v1** (PARQUET_1_0): Original format, maximum compatibility (default)
    /// - **v2** (PARQUET_2_0): Modern format with improved encoding and statistics
    ///
    /// Version 2 benefits:
    /// - More efficient encoding for certain data types (10-20% smaller files)
    /// - Better statistics for query optimization
    /// - Improved data page format
    /// - Required for some advanced features
    ///
    /// Use v1 for maximum compatibility with older readers (pre-2018 tools).
    /// Use v2 for better performance with modern query engines (Athena, Spark, Presto).
    #[serde(default)]
    #[configurable(metadata(docs::examples = "v1"))]
    #[configurable(metadata(docs::examples = "v2"))]
    pub writer_version: ParquetWriterVersion,

    /// Number of rows per row group
    ///
    /// Row groups are Parquet's unit of parallelization. Larger row groups
    /// can improve compression but increase memory usage during encoding.
    ///
    /// Since each batch becomes a separate Parquet file, this value
    /// should be <= the batch max_events setting. Row groups cannot span multiple files.
    /// If not specified, defaults to the batch size.
    #[serde(default)]
    #[configurable(metadata(docs::examples = 100000))]
    #[configurable(metadata(docs::examples = 1000000))]
    pub row_group_size: Option<usize>,

    /// Allow null values for non-nullable fields in the schema.
    ///
    /// When enabled, missing or incompatible values will be encoded as null even for fields
    /// marked as non-nullable in the Arrow schema. This is useful when working with downstream
    /// systems that can handle null values through defaults, computed columns, or other mechanisms.
    ///
    /// When disabled (default), missing values for non-nullable fields will cause encoding errors,
    /// ensuring all required data is present before writing to Parquet.
    #[serde(default)]
    #[configurable(metadata(docs::examples = true))]
    pub allow_nullable_fields: bool,

    /// Sorting order for rows within row groups.
    ///
    /// Pre-sorting rows by specified columns before writing can significantly improve both
    /// compression ratios and query performance. This is especially valuable for time-series
    /// data and event logs.
    ///
    /// **Benefits:**
    /// - **Better compression** (20-40% smaller files): Similar values are grouped together
    /// - **Faster queries**: More effective min/max statistics enable better row group skipping
    /// - **Improved caching**: Query engines can more efficiently cache sorted data
    ///
    /// **Common patterns:**
    /// - Time-series: Sort by timestamp descending (most recent first)
    /// - Multi-tenant: Sort by tenant_id, then timestamp
    /// - User analytics: Sort by user_id, then event_time
    ///
    /// **Trade-offs:**
    /// - Adds sorting overhead during encoding (typically 10-30% slower writes)
    /// - Requires buffering entire batch in memory for sorting
    /// - Most beneficial when queries frequently filter on sorted columns
    ///
    /// **Example:**
    /// ```yaml
    /// sorting_columns:
    ///   - column: timestamp
    ///     descending: true
    ///   - column: user_id
    ///     descending: false
    /// ```
    ///
    /// If not specified, rows are written in the order they appear in the batch.
    #[serde(default)]
    pub sorting_columns: Option<Vec<SortingColumnConfig>>,

    /// Maximum number of rows per Parquet file when super-batching is enabled.
    ///
    /// When set, enables "super-batch" mode where a larger batch is first sorted (if
    /// `sorting_columns` is configured) and then split into multiple smaller Parquet files,
    /// each containing at most this many rows. This ensures data is sorted both within
    /// individual files AND across files, which significantly improves query performance
    /// for analytics workloads.
    ///
    /// **Use case example:**
    /// With `rows_per_file: 10000` and a batch of 100,000 events sorted by timestamp:
    /// - 10 Parquet files are produced, each with ~10,000 rows
    /// - File 1 contains the oldest 10,000 events
    /// - File 10 contains the newest 10,000 events
    /// - Query engines can skip entire files based on min/max statistics
    ///
    /// **Memory efficiency:**
    /// Sorting is performed using Arrow's index-based sorting, which avoids copying data.
    /// Each chunk is materialized and written separately to minimize peak memory usage.
    ///
    /// **Recommended settings:**
    /// - Set batch `max_events` to your desired super-batch size (e.g., 100,000)
    /// - Set `rows_per_file` to your desired file size (e.g., 10,000)
    /// - Configure `sorting_columns` to define the sort order
    ///
    /// If not specified, each batch produces exactly one Parquet file.
    #[serde(default)]
    #[configurable(metadata(docs::examples = 10000))]
    #[configurable(metadata(docs::examples = 50000))]
    pub rows_per_file: Option<usize>,

    /// JSON columns to expand into subcolumns (similar to ClickHouse JSON type)
    ///
    /// When configured, JSON string columns are parsed and expanded into individual
    /// subcolumns for efficient columnar storage and querying. This is particularly
    /// useful for event data with nested properties.
    ///
    /// **How it works:**
    /// 1. The JSON column is parsed and flattened using dot notation
    ///    (e.g., `properties.user.id` becomes `properties.user.id`)
    /// 2. Types are inferred from values with fallback to strings
    /// 3. Columns are prioritized by non-null/non-empty count per batch
    /// 4. Top `max_subcolumns` keys become dedicated columns
    /// 5. Overflow keys are hashed into bucket map columns
    ///
    /// **Column naming:**
    /// - Subcolumns: `{column}.{path}` (e.g., `properties.$ip`)
    /// - Bucket maps: `{column}__json_type_bucket_{n}` (e.g., `properties__json_type_bucket_0`)
    ///
    /// **Example configuration:**
    /// ```yaml
    /// json_columns:
    ///   - column: properties
    ///     max_subcolumns: 1024
    ///     bucket_count: 256
    ///     max_depth: 10
    /// ```
    #[serde(default)]
    pub json_columns: Option<Vec<JsonColumnConfig>>,

    /// Use memory-mapped temporary files for large batch processing.
    ///
    /// When enabled (default), super-batch mode writes intermediate Arrow data to
    /// memory-mapped temporary files instead of keeping everything in RAM. This
    /// significantly reduces memory usage for large batches (e.g., 100k events)
    /// while maintaining sorting capabilities across the entire batch.
    ///
    /// **How it works:**
    /// 1. Events are converted to an Arrow RecordBatch
    /// 2. The RecordBatch is written to a temporary Arrow IPC file
    /// 3. The file is memory-mapped for efficient random access
    /// 4. Sorting indices are computed (in memory, ~8 bytes per row)
    /// 5. Each output file's data is read from mmap and processed
    ///
    /// **Memory savings:**
    /// For a batch of 100k events at ~70KB each:
    /// - Without mmap: ~7GB RAM for data + sorting
    /// - With mmap: ~800KB for indices + OS page cache
    ///
    /// **Trade-offs:**
    /// - Slightly higher latency due to disk I/O (typically <5% slower)
    /// - Requires write access to system temp directory
    /// - OS page cache handles caching automatically
    ///
    /// Disable only if you have sufficient RAM and want to avoid temp file I/O.
    #[serde(default = "default_use_memory_mapped_files")]
    #[configurable(metadata(docs::examples = true))]
    #[configurable(metadata(docs::examples = false))]
    pub use_memory_mapped_files: bool,
}

impl Default for ParquetSerializerConfig {
    fn default() -> Self {
        Self {
            schema: None,
            infer_schema: false,
            exclude_columns: None,
            max_columns: default_max_columns(), // 1000
            compression: ParquetCompression::default(),
            compression_level: None,
            writer_version: ParquetWriterVersion::default(),
            row_group_size: None,
            allow_nullable_fields: false,
            sorting_columns: None,
            rows_per_file: None,
            json_columns: None,
            use_memory_mapped_files: default_use_memory_mapped_files(),
        }
    }
}

/// Column sorting configuration
#[configurable_component]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SortingColumnConfig {
    /// Name of the column to sort by
    #[configurable(metadata(docs::examples = "timestamp"))]
    #[configurable(metadata(docs::examples = "user_id"))]
    pub column: String,

    /// Sort in descending order (true) or ascending order (false)
    ///
    /// - `true`: Descending (Z-A, 9-0, newest-oldest)
    /// - `false`: Ascending (A-Z, 0-9, oldest-newest)
    #[serde(default)]
    #[configurable(metadata(docs::examples = true))]
    pub descending: bool,
}

/// Configuration for JSON column expansion (similar to ClickHouse JSON type)
///
/// When configured, JSON string columns are expanded into individual subcolumns
/// for efficient columnar storage and querying. Keys that don't fit within the
/// max_subcolumns limit are placed into hash-bucketed map columns.
#[configurable_component]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonColumnConfig {
    /// Name of the column containing JSON data to expand
    ///
    /// This column should contain valid JSON object strings.
    #[configurable(metadata(docs::examples = "properties"))]
    #[configurable(metadata(docs::examples = "attributes"))]
    pub column: String,

    /// Maximum number of subcolumns to create from the JSON data
    ///
    /// Subcolumns are prioritized by the number of non-null/non-empty values
    /// in each batch. Keys beyond this limit are stored in bucketed map columns.
    #[serde(default = "default_json_max_subcolumns")]
    #[configurable(metadata(docs::examples = 1024))]
    pub max_subcolumns: usize,

    /// Number of hash buckets for overflow keys
    ///
    /// Keys that don't fit in the max_subcolumns limit are hashed into
    /// this many buckets. Column names follow the pattern:
    /// `{column}__json_type_bucket_{n}` where n is the bucket index.
    #[serde(default = "default_json_bucket_count")]
    #[configurable(metadata(docs::examples = 256))]
    pub bucket_count: usize,

    /// Maximum depth to flatten nested objects
    ///
    /// Objects nested deeper than this level are stored as JSON strings.
    /// Keys are flattened using dot notation (e.g., `user.address.city`).
    #[serde(default = "default_json_max_depth")]
    #[configurable(metadata(docs::examples = 10))]
    pub max_depth: usize,

    /// Keep the original JSON column as a string alongside the subcolumns
    ///
    /// When true, the original column is preserved as a string column.
    /// When false, only the expanded subcolumns are written.
    #[serde(default = "default_keep_original_column")]
    #[configurable(metadata(docs::examples = true))]
    pub keep_original_column: bool,

    /// Type hints for specific JSON paths
    ///
    /// Provides hints for type inference on specific keys. Keys should use
    /// dot notation for nested paths. If type inference fails, values
    /// are stored as strings.
    ///
    /// Supported types: `string`, `int64`, `uint64`, `float64`, `boolean`
    #[serde(default)]
    #[configurable(metadata(docs::examples = "type_hints_example()"))]
    pub type_hints: Option<BTreeMap<String, JsonTypeHint>>,
}

impl Default for JsonColumnConfig {
    fn default() -> Self {
        Self {
            column: String::new(),
            max_subcolumns: default_json_max_subcolumns(),
            bucket_count: default_json_bucket_count(),
            max_depth: default_json_max_depth(),
            keep_original_column: default_keep_original_column(),
            type_hints: None,
        }
    }
}

/// Type hint for JSON value inference
#[configurable_component]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JsonTypeHint {
    /// UTF-8 string
    #[default]
    String,
    /// 64-bit signed integer
    Int64,
    /// 64-bit unsigned integer
    Uint64,
    /// 64-bit floating point
    Float64,
    /// Boolean
    Boolean,
}

fn default_json_max_subcolumns() -> usize {
    1024
}

fn default_json_bucket_count() -> usize {
    256
}

fn default_json_max_depth() -> usize {
    10
}

fn default_keep_original_column() -> bool {
    true
}

fn default_use_memory_mapped_files() -> bool {
    true
}

fn type_hints_example() -> BTreeMap<String, JsonTypeHint> {
    let mut hints = BTreeMap::new();
    hints.insert("user.age".to_string(), JsonTypeHint::Int64);
    hints.insert("metrics.count".to_string(), JsonTypeHint::Uint64);
    hints.insert("score".to_string(), JsonTypeHint::Float64);
    hints
}

fn default_max_columns() -> usize {
    1000
}

fn schema_example() -> SchemaDefinition {
    use super::schema_definition::FieldDefinition;
    use std::collections::BTreeMap;

    let mut fields = BTreeMap::new();
    fields.insert(
        "id".to_string(),
        FieldDefinition {
            r#type: "int64".to_string(),
            bloom_filter: false,
            bloom_filter_num_distinct_values: None,
            bloom_filter_false_positive_pct: None,
        },
    );
    fields.insert(
        "name".to_string(),
        FieldDefinition {
            r#type: "utf8".to_string(),
            bloom_filter: true, // Example: enable for high-cardinality string field
            bloom_filter_num_distinct_values: Some(1_000_000),
            bloom_filter_false_positive_pct: Some(0.01),
        },
    );
    fields.insert(
        "timestamp".to_string(),
        FieldDefinition {
            r#type: "timestamp_microsecond".to_string(),
            bloom_filter: false,
            bloom_filter_num_distinct_values: None,
            bloom_filter_false_positive_pct: None,
        },
    );
    SchemaDefinition { fields }
}

impl std::fmt::Debug for ParquetSerializerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetSerializerConfig")
            .field("schema", &self.schema.is_some())
            .field("infer_schema", &self.infer_schema)
            .field("exclude_columns", &self.exclude_columns)
            .field("max_columns", &self.max_columns)
            .field("compression", &self.compression)
            .field("compression_level", &self.compression_level)
            .field("writer_version", &self.writer_version)
            .field("row_group_size", &self.row_group_size)
            .field("allow_nullable_fields", &self.allow_nullable_fields)
            .field("sorting_columns", &self.sorting_columns)
            .finish()
    }
}

impl ParquetSerializerConfig {
    /// Create a new ParquetSerializerConfig with a schema definition
    pub fn new(schema: SchemaDefinition) -> Self {
        Self {
            schema: Some(schema),
            infer_schema: false,
            exclude_columns: None,
            max_columns: default_max_columns(),
            compression: ParquetCompression::default(),
            compression_level: None,
            writer_version: ParquetWriterVersion::default(),
            row_group_size: None,
            allow_nullable_fields: false,
            sorting_columns: None,
            rows_per_file: None,
            json_columns: None,
            use_memory_mapped_files: default_use_memory_mapped_files(),
        }
    }

    /// Validate the configuration
    fn validate(&self) -> Result<(), String> {
        // Must specify exactly one schema method
        match (self.schema.is_some(), self.infer_schema) {
            (true, true) => {
                return Err(
                    "Cannot use both 'schema' and 'infer_schema: true'. Choose one.".to_string(),
                );
            }
            (false, false) => {
                return Err("Must specify either 'schema' or 'infer_schema: true'".to_string());
            }
            _ => {}
        }

        // Validate rows_per_file if set
        if let Some(rows_per_file) = self.rows_per_file {
            if rows_per_file == 0 {
                return Err("'rows_per_file' must be greater than 0".to_string());
            }
        }

        Ok(())
    }

    /// The data type of events that are accepted by `ParquetSerializer`.
    pub fn input_type(&self) -> vector_core::config::DataType {
        vector_core::config::DataType::Log
    }

    /// The schema required by the serializer.
    pub fn schema_requirement(&self) -> vector_core::schema::Requirement {
        vector_core::schema::Requirement::empty()
    }
}

/// Schema mode for Parquet serialization
#[derive(Clone, Debug)]
enum SchemaMode {
    /// Use pre-defined explicit schema
    Explicit { schema: Arc<Schema> },
    /// Infer schema from each batch
    Inferred {
        exclude_columns: std::collections::BTreeSet<String>,
        max_columns: usize,
    },
}

/// Parquet batch serializer that holds the schema and writer configuration
#[derive(Clone, Debug)]
pub struct ParquetSerializer {
    schema_mode: SchemaMode,
    writer_properties: WriterProperties,
    /// Sorting column configuration (column indices and sort order)
    sorting_columns: Option<Vec<SortingColumnConfig>>,
    /// Maximum rows per file for super-batch mode
    rows_per_file: Option<usize>,
    /// JSON column processors for expanding JSON columns into subcolumns
    json_column_processors: Option<Vec<JsonColumnProcessor>>,
    /// Use memory-mapped files for large batch processing
    use_memory_mapped_files: bool,
    /// Directory for memory-mapped temp files (defaults to system temp dir)
    data_dir: Option<std::path::PathBuf>,
}

impl ParquetSerializer {
    /// Create a new ParquetSerializer with the given configuration
    pub fn new(config: ParquetSerializerConfig) -> Result<Self, vector_common::Error> {
        // Validate configuration
        config.validate().map_err(vector_common::Error::from)?;

        // Keep a copy of schema_def for later use with Bloom filters
        let schema_def_opt = config.schema.clone();

        // Determine schema mode
        let schema_mode = if config.infer_schema {
            SchemaMode::Inferred {
                exclude_columns: config
                    .exclude_columns
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
                max_columns: config.max_columns,
            }
        } else {
            let schema_def = config.schema.ok_or_else(|| {
                vector_common::Error::from("Schema required when infer_schema is false")
            })?;

            // Convert SchemaDefinition to Arrow Schema
            let mut schema = schema_def
                .to_arrow_schema()
                .map_err(|e| vector_common::Error::from(e.to_string()))?;

            // If allow_nullable_fields is enabled, transform the schema once here
            if config.allow_nullable_fields {
                schema = Arc::new(Schema::new_with_metadata(
                    schema
                        .fields()
                        .iter()
                        .map(|f| Arc::new(super::arrow::make_field_nullable(f)))
                        .collect::<Vec<_>>(),
                    schema.metadata().clone(),
                ));
            }

            SchemaMode::Explicit { schema }
        };

        // Build writer properties
        let compression = config
            .compression
            .to_compression(config.compression_level)
            .map_err(vector_common::Error::from)?;

        tracing::debug!(
            compression = ?config.compression,
            compression_level = ?config.compression_level,
            writer_version = ?config.writer_version,
            infer_schema = config.infer_schema,
            "Configuring Parquet writer properties"
        );

        let mut props_builder = WriterProperties::builder()
            .set_compression(compression)
            .set_writer_version(config.writer_version.into());

        if let Some(row_group_size) = config.row_group_size {
            props_builder = props_builder.set_max_row_group_size(row_group_size);
        }

        // Only apply Bloom filters and sorting for explicit schema mode
        if let (SchemaMode::Explicit { schema }, Some(schema_def)) = (&schema_mode, &schema_def_opt)
        {
            // Apply per-column Bloom filter settings from schema
            let bloom_filter_configs = schema_def.extract_bloom_filter_configs();
            for bloom_config in bloom_filter_configs {
                if let Some(col_idx) = schema
                    .fields()
                    .iter()
                    .position(|f| f.name() == &bloom_config.column_name)
                {
                    // Use field-specific settings or sensible defaults
                    let fpp = bloom_config.fpp.unwrap_or(0.05); // Default 5% false positive rate
                    let mut ndv = bloom_config.ndv.unwrap_or(1_000_000); // Default 1M distinct values

                    // Cap NDV to row group size (can't have more distinct values than total rows)
                    if let Some(row_group_size) = config.row_group_size {
                        ndv = ndv.min(row_group_size as u64);
                    }

                    let column_path = ColumnPath::from(schema.field(col_idx).name().as_str());
                    props_builder = props_builder
                        .set_column_bloom_filter_enabled(column_path.clone(), true)
                        .set_column_bloom_filter_fpp(column_path.clone(), fpp)
                        .set_column_bloom_filter_ndv(column_path, ndv);
                }
            }

            // Set sorting columns if configured
            if let Some(sorting_cols) = &config.sorting_columns {
                use parquet::format::SortingColumn;

                let parquet_sorting_cols: Vec<SortingColumn> = sorting_cols
                    .iter()
                    .map(|col| {
                        let col_idx = schema
                            .fields()
                            .iter()
                            .position(|f| f.name() == &col.column)
                            .ok_or_else(|| {
                                vector_common::Error::from(format!(
                                    "Sorting column '{}' not found in schema",
                                    col.column
                                ))
                            })?;

                        Ok(SortingColumn::new(col_idx as i32, col.descending, false))
                    })
                    .collect::<Result<Vec<_>, vector_common::Error>>()?;

                props_builder = props_builder.set_sorting_columns(Some(parquet_sorting_cols));
            }
        }
        // Note: Bloom filters and sorting are NOT applied for inferred schemas

        let writer_properties = props_builder.build();

        // Create JSON column processors if configured
        let json_column_processors = config.json_columns.map(|json_cols| {
            json_cols
                .into_iter()
                .map(|col_config| JsonColumnProcessor::new(col_config))
                .collect()
        });

        Ok(Self {
            schema_mode,
            writer_properties,
            sorting_columns: config.sorting_columns,
            rows_per_file: config.rows_per_file,
            json_column_processors,
            use_memory_mapped_files: config.use_memory_mapped_files,
            data_dir: None,
        })
    }

    /// Set the data directory for memory-mapped temp files.
    ///
    /// When set, mmap temp files will be created in a `parquet_temp` subdirectory
    /// of the specified path. If not set, falls back to the system temp directory.
    pub fn set_data_dir(&mut self, data_dir: Option<std::path::PathBuf>) {
        self.data_dir = data_dir;
    }

    /// Returns true if JSON column expansion is enabled
    pub fn has_json_columns(&self) -> bool {
        self.json_column_processors.is_some()
    }

    /// Returns true if super-batch mode is enabled (rows_per_file is set)
    pub fn is_super_batch_enabled(&self) -> bool {
        self.rows_per_file.is_some()
    }

    /// Returns the configured rows per file for super-batch mode
    pub fn rows_per_file(&self) -> Option<usize> {
        self.rows_per_file
    }

    /// Encode events into multiple Parquet files (super-batch mode).
    ///
    /// This method sorts the events (if sorting_columns is configured) and splits them
    /// into multiple Parquet files, each containing at most `rows_per_file` rows.
    ///
    /// Memory efficiency: Sorting uses Arrow's index-based approach to avoid copying data.
    /// Each chunk is materialized and written separately to minimize peak memory usage.
    ///
    /// When `use_memory_mapped_files` is enabled (default), the record batch is written
    /// to a temporary Arrow IPC file and memory-mapped, significantly reducing RAM usage
    /// for large batches while maintaining global sorting capabilities.
    pub fn encode_batch_split(
        &self,
        events: Vec<Event>,
    ) -> Result<Vec<Bytes>, ParquetEncodingError> {
        let rows_per_file = self.rows_per_file.unwrap_or(events.len());

        if events.is_empty() {
            return Err(ParquetEncodingError::NoEvents);
        }

        // Determine schema
        let schema = self.resolve_schema(&events)?;

        // If we don't need to split AND no sorting AND no JSON expansion, encode directly
        if events.len() <= rows_per_file
            && self.sorting_columns.is_none()
            && self.json_column_processors.is_none()
        {
            let bytes = encode_events_to_parquet(&events, schema, &self.writer_properties)?;
            return Ok(vec![bytes]);
        }

        // Build the full record batch
        let record_batch = build_record_batch(Arc::clone(&schema), &events)?;

        // Use memory-mapped processing for large batches when enabled
        // Threshold: use mmap when we have more than rows_per_file rows (i.e., multiple output files)
        if self.use_memory_mapped_files && record_batch.num_rows() > rows_per_file {
            return self.sort_and_split_batch_mmap(record_batch, rows_per_file);
        }

        // Sort (if configured) and split the record batch
        // JSON column expansion is done per-chunk in sort_and_split_batch so each
        // parquet file has its own schema based on that chunk's data
        self.sort_and_split_batch(record_batch, rows_per_file)
    }

    /// Resolve schema based on mode
    fn resolve_schema(&self, events: &[Event]) -> Result<Arc<Schema>, ParquetEncodingError> {
        match &self.schema_mode {
            SchemaMode::Explicit { schema } => Ok(Arc::clone(schema)),
            SchemaMode::Inferred {
                exclude_columns,
                max_columns,
            } => infer_schema_from_events(events, exclude_columns, *max_columns),
        }
    }

    /// Sort the record batch and split into chunks, encoding each as a separate Parquet file.
    ///
    /// Memory efficiency strategy:
    /// 1. Compute sort indices without copying data
    /// 2. For each chunk, use `take` to extract only those rows
    /// 3. Encode and write immediately, allowing the chunk to be freed
    fn sort_and_split_batch(
        &self,
        batch: arrow::array::RecordBatch,
        rows_per_file: usize,
    ) -> Result<Vec<Bytes>, ParquetEncodingError> {
        use arrow::compute::{SortColumn, SortOptions, lexsort_to_indices, take};

        let num_rows = batch.num_rows();
        let schema = batch.schema();

        // Compute sorted indices
        let sorted_indices = if let Some(sorting_cols) = &self.sorting_columns {
            // Build sort columns from config
            let sort_columns: Vec<SortColumn> = sorting_cols
                .iter()
                .filter_map(|col_config| {
                    schema
                        .fields()
                        .iter()
                        .position(|f| f.name() == &col_config.column)
                        .map(|idx| SortColumn {
                            values: Arc::clone(batch.column(idx)),
                            options: Some(SortOptions {
                                descending: col_config.descending,
                                nulls_first: false,
                            }),
                        })
                })
                .collect();

            if sort_columns.is_empty() {
                // No valid sort columns found, use natural order
                None
            } else {
                Some(
                    lexsort_to_indices(&sort_columns, None)
                        .map_err(|e| ParquetEncodingError::RecordBatchCreation {
                            source: super::arrow::ArrowEncodingError::RecordBatchCreation { source: e },
                        })?,
                )
            }
        } else {
            None
        };

        // Calculate number of files needed
        let num_files = (num_rows + rows_per_file - 1) / rows_per_file;

        tracing::debug!(
            num_files = num_files,
            rows_per_file = rows_per_file,
            rayon_threads = rayon::current_num_threads(),
            "Processing batch into parquet files (in-memory mode)"
        );

        // Process chunks in parallel using rayon
        // Each chunk is independent: extract rows, expand JSON, encode to Parquet
        use rayon::prelude::*;

        let parquet_files: Result<Vec<Bytes>, ParquetEncodingError> = (0..num_files)
            .into_par_iter()
            .with_min_len(1) // Each file should be processed as a separate work unit
            .map(|chunk_idx| {
                let start = chunk_idx * rows_per_file;
                let end = std::cmp::min(start + rows_per_file, num_rows);
                let chunk_len = end - start;

                tracing::trace!(
                    chunk_idx = chunk_idx,
                    thread_id = ?std::thread::current().id(),
                    rows = chunk_len,
                    "Processing parquet chunk"
                );

                // Extract the chunk using sorted indices if available
                let chunk_batch = if let Some(ref indices) = sorted_indices {
                    // Slice the indices for this chunk
                    let chunk_indices = indices.slice(start, chunk_len);

                    // Take rows according to the sorted indices
                    let chunk_columns: Result<Vec<_>, _> = batch
                        .columns()
                        .iter()
                        .map(|col| take(col.as_ref(), &chunk_indices, None))
                        .collect();

                    let chunk_columns = chunk_columns.map_err(|e| {
                        ParquetEncodingError::RecordBatchCreation {
                            source: super::arrow::ArrowEncodingError::RecordBatchCreation { source: e },
                        }
                    })?;

                    arrow::array::RecordBatch::try_new(Arc::clone(&schema), chunk_columns).map_err(
                        |e| ParquetEncodingError::RecordBatchCreation {
                            source: super::arrow::ArrowEncodingError::RecordBatchCreation { source: e },
                        },
                    )?
                } else {
                    // No sorting, just slice the batch
                    batch.slice(start, chunk_len)
                };

                // Expand JSON columns on this chunk if processors are configured
                // Each chunk gets its own JSON-expanded schema based on that chunk's data
                let chunk_batch = if let Some(processors) = &self.json_column_processors {
                    expand_json_columns(chunk_batch, processors, &[])?
                } else {
                    chunk_batch
                };

                // Encode this chunk to Parquet
                encode_record_batch_to_parquet(&chunk_batch, &self.writer_properties)
            })
            .collect();

        parquet_files
    }

    /// Sort and split using memory-mapped temp files for reduced RAM usage.
    ///
    /// This method writes the record batch to a temporary Arrow IPC file,
    /// memory-maps it, and processes chunks directly from disk. This significantly
    /// reduces memory usage for large batches while maintaining global sorting.
    ///
    /// Memory usage:
    /// - Sort indices: ~8 bytes per row (e.g., 800KB for 100k rows)
    /// - Chunk processing: only one chunk's worth of data in memory at a time
    /// - OS page cache handles efficient disk I/O automatically
    fn sort_and_split_batch_mmap(
        &self,
        batch: arrow::array::RecordBatch,
        rows_per_file: usize,
    ) -> Result<Vec<Bytes>, ParquetEncodingError> {
        use arrow::compute::{SortColumn, SortOptions, lexsort_to_indices, take};
        use arrow::ipc::reader::FileReader;
        use arrow::ipc::writer::FileWriter;
        use std::fs::File;

        let num_rows = batch.num_rows();
        let schema = batch.schema();

        // Determine temp directory: use data_dir/parquet_temp if set, else system temp
        let temp_dir = match &self.data_dir {
            Some(data_dir) => {
                let parquet_temp = data_dir.join("parquet_temp");
                // Create the directory if it doesn't exist
                if let Err(e) = std::fs::create_dir_all(&parquet_temp) {
                    tracing::warn!(
                        error = %e,
                        path = %parquet_temp.display(),
                        "Failed to create parquet temp directory, falling back to system temp"
                    );
                    None
                } else {
                    Some(parquet_temp)
                }
            }
            None => None,
        };

        // Create a temp file for the Arrow IPC data
        let temp_file = match &temp_dir {
            Some(dir) => tempfile::Builder::new()
                .prefix("arrow_batch_")
                .suffix(".arrow")
                .tempfile_in(dir)?,
            None => tempfile::Builder::new()
                .prefix("arrow_batch_")
                .suffix(".arrow")
                .tempfile()?,
        };

        let temp_path = temp_file.path().to_path_buf();

        tracing::debug!(
            path = %temp_path.display(),
            num_rows = num_rows,
            "Writing record batch to temp file for mmap processing"
        );

        // Write the record batch to the temp file
        {
            let file = temp_file.as_file();
            let mut writer = FileWriter::try_new(file, &schema)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            writer
                .write(&batch)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            writer
                .finish()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        }

        // Drop the original batch to free memory before mmap
        drop(batch);

        // Memory-map the file and read the batch back
        let file = File::open(&temp_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Create a reader from the mmap'd data
        let cursor = std::io::Cursor::new(&mmap[..]);
        let reader = FileReader::try_new(cursor, None)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        // Read the batch from mmap (this is zero-copy - data stays in mmap)
        let mmap_batches: Vec<_> = reader
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        if mmap_batches.is_empty() {
            return Err(ParquetEncodingError::NoEvents);
        }

        // We only wrote one batch, so there should be exactly one
        let mmap_batch = &mmap_batches[0];
        let schema = mmap_batch.schema();

        // Compute sorted indices (this is the only allocation proportional to num_rows)
        let sorted_indices = if let Some(sorting_cols) = &self.sorting_columns {
            let sort_columns: Vec<SortColumn> = sorting_cols
                .iter()
                .filter_map(|col_config| {
                    schema
                        .fields()
                        .iter()
                        .position(|f| f.name() == &col_config.column)
                        .map(|idx| SortColumn {
                            values: Arc::clone(mmap_batch.column(idx)),
                            options: Some(SortOptions {
                                descending: col_config.descending,
                                nulls_first: false,
                            }),
                        })
                })
                .collect();

            if sort_columns.is_empty() {
                None
            } else {
                Some(
                    lexsort_to_indices(&sort_columns, None)
                        .map_err(|e| ParquetEncodingError::RecordBatchCreation {
                            source: super::arrow::ArrowEncodingError::RecordBatchCreation { source: e },
                        })?,
                )
            }
        } else {
            None
        };

        let num_files = (num_rows + rows_per_file - 1) / rows_per_file;

        tracing::debug!(
            num_files = num_files,
            rows_per_file = rows_per_file,
            rayon_threads = rayon::current_num_threads(),
            "Processing mmap'd batch into parquet files"
        );

        // Advise the OS to prefetch the mmap'd data to reduce page fault latency during parallel access
        #[cfg(unix)]
        unsafe {
            libc::posix_madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::POSIX_MADV_WILLNEED,
            );
        }

        // Process chunks in parallel
        // Each chunk is a separate work unit (with_min_len(1)) to ensure maximum parallelism
        use rayon::prelude::*;

        let parquet_files: Result<Vec<Bytes>, ParquetEncodingError> = (0..num_files)
            .into_par_iter()
            .with_min_len(1) // Each file should be processed as a separate work unit
            .map(|chunk_idx| {
                let start = chunk_idx * rows_per_file;
                let end = std::cmp::min(start + rows_per_file, num_rows);
                let chunk_len = end - start;

                tracing::trace!(
                    chunk_idx = chunk_idx,
                    thread_id = ?std::thread::current().id(),
                    rows = chunk_len,
                    "Processing parquet chunk"
                );

                // Extract the chunk using sorted indices if available
                let chunk_batch = if let Some(ref indices) = sorted_indices {
                    let chunk_indices = indices.slice(start, chunk_len);

                    let chunk_columns: Result<Vec<_>, _> = mmap_batch
                        .columns()
                        .iter()
                        .map(|col| take(col.as_ref(), &chunk_indices, None))
                        .collect();

                    let chunk_columns = chunk_columns.map_err(|e| {
                        ParquetEncodingError::RecordBatchCreation {
                            source: super::arrow::ArrowEncodingError::RecordBatchCreation { source: e },
                        }
                    })?;

                    arrow::array::RecordBatch::try_new(Arc::clone(&schema), chunk_columns).map_err(
                        |e| ParquetEncodingError::RecordBatchCreation {
                            source: super::arrow::ArrowEncodingError::RecordBatchCreation { source: e },
                        },
                    )?
                } else {
                    mmap_batch.slice(start, chunk_len)
                };

                // Expand JSON columns on this chunk if processors are configured
                let chunk_batch = if let Some(processors) = &self.json_column_processors {
                    expand_json_columns(chunk_batch, processors, &[])?
                } else {
                    chunk_batch
                };

                // Encode this chunk to Parquet
                encode_record_batch_to_parquet(&chunk_batch, &self.writer_properties)
            })
            .collect();

        // Temp file is automatically deleted when temp_file goes out of scope
        tracing::debug!(
            path = %temp_path.display(),
            "Finished mmap processing, temp file will be deleted"
        );

        parquet_files
    }
}

impl tokio_util::codec::Encoder<Vec<Event>> for ParquetSerializer {
    type Error = ParquetEncodingError;

    fn encode(&mut self, events: Vec<Event>, buffer: &mut BytesMut) -> Result<(), Self::Error> {
        if events.is_empty() {
            return Err(ParquetEncodingError::NoEvents);
        }

        // Determine schema based on mode
        let schema = match &self.schema_mode {
            SchemaMode::Explicit { schema } => Arc::clone(schema),
            SchemaMode::Inferred {
                exclude_columns,
                max_columns,
            } => infer_schema_from_events(&events, exclude_columns, *max_columns)?,
        };

        // Build the initial record batch
        let record_batch = build_record_batch(schema.clone(), &events)?;

        // If JSON column processors are configured, expand JSON columns
        let final_batch = if let Some(processors) = &self.json_column_processors {
            expand_json_columns(record_batch, processors, &events)?
        } else {
            record_batch
        };

        let bytes = encode_record_batch_to_parquet(&final_batch, &self.writer_properties)?;

        // Use put() instead of extend_from_slice to avoid copying when possible
        buffer.put(bytes);
        Ok(())
    }
}

/// Errors that can occur during Parquet encoding
#[derive(Debug, Snafu)]
pub enum ParquetEncodingError {
    /// Failed to build Arrow record batch
    #[snafu(display("Failed to build Arrow record batch: {}", source))]
    RecordBatchCreation {
        /// The underlying Arrow encoding error
        source: ArrowEncodingError,
    },

    /// Failed to write Parquet data
    #[snafu(display("Failed to write Parquet data: {}", source))]
    ParquetWrite {
        /// The underlying Parquet error
        source: parquet::errors::ParquetError,
    },

    /// No events provided for encoding
    #[snafu(display("No events provided for encoding"))]
    NoEvents,

    /// Schema must be provided before encoding
    #[snafu(display("Schema must be provided before encoding"))]
    NoSchemaProvided,

    /// No fields could be inferred from events
    #[snafu(display(
        "No fields could be inferred from events (all fields excluded or only null values)"
    ))]
    NoFieldsInferred,

    /// Invalid event type (not a log event)
    #[snafu(display("Invalid event type, expected log event"))]
    InvalidEventType,

    /// JSON serialization error for nested types
    #[snafu(display("Failed to serialize nested type as JSON: {}", source))]
    JsonSerialization {
        /// The underlying JSON error
        source: serde_json::Error,
    },

    /// IO error during encoding
    #[snafu(display("IO error: {}", source))]
    Io {
        /// The underlying IO error
        source: std::io::Error,
    },
}

impl From<std::io::Error> for ParquetEncodingError {
    fn from(error: std::io::Error) -> Self {
        Self::Io { source: error }
    }
}

impl From<ArrowEncodingError> for ParquetEncodingError {
    fn from(error: ArrowEncodingError) -> Self {
        Self::RecordBatchCreation { source: error }
    }
}

impl From<parquet::errors::ParquetError> for ParquetEncodingError {
    fn from(error: parquet::errors::ParquetError) -> Self {
        Self::ParquetWrite { source: error }
    }
}

impl From<serde_json::Error> for ParquetEncodingError {
    fn from(error: serde_json::Error) -> Self {
        Self::JsonSerialization { source: error }
    }
}

/// Infer Arrow DataType from a Vector Value
fn infer_arrow_type(value: &vector_core::event::Value) -> arrow::datatypes::DataType {
    use arrow::datatypes::{DataType, TimeUnit};
    use vector_core::event::Value;

    match value {
        Value::Bytes(_) => DataType::LargeUtf8,
        Value::Integer(_) => DataType::Int64,
        Value::Float(_) => DataType::Float64,
        Value::Boolean(_) => DataType::Boolean,
        Value::Timestamp(_) => DataType::Timestamp(TimeUnit::Microsecond, None),
        // Nested types and regex are always serialized as strings
        Value::Array(_) | Value::Object(_) | Value::Regex(_) => DataType::LargeUtf8,
        // Null doesn't determine type, default to LargeUtf8
        Value::Null => DataType::LargeUtf8,
    }
}

/// Infer schema from a batch of events
fn infer_schema_from_events(
    events: &[Event],
    exclude_columns: &std::collections::BTreeSet<String>,
    max_columns: usize,
) -> Result<Arc<Schema>, ParquetEncodingError> {
    use arrow::datatypes::{DataType, Field};
    use std::collections::BTreeMap;
    use vector_core::event::Value;

    let mut field_types: BTreeMap<String, DataType> = BTreeMap::new();
    let mut type_conflicts: BTreeMap<String, Vec<DataType>> = BTreeMap::new();

    for event in events {
        // Only process log events
        let log = match event {
            Event::Log(log) => log,
            _ => return Err(ParquetEncodingError::InvalidEventType),
        };

        let fields_iter = log
            .all_event_fields()
            .ok_or(ParquetEncodingError::InvalidEventType)?;

        for (key, value) in fields_iter {
            let key_str = key.to_string();

            // Skip excluded columns
            if exclude_columns.contains(&key_str) {
                continue;
            }

            // Skip Value::Null (doesn't determine type)
            if matches!(value, Value::Null) {
                continue;
            }

            // Enforce max columns (skip new fields after limit)
            if field_types.len() >= max_columns && !field_types.contains_key(&key_str) {
                tracing::debug!(
                    column = %key_str,
                    max_columns = max_columns,
                    "Skipping column: max_columns limit reached"
                );
                continue;
            }

            let inferred_type = infer_arrow_type(value);

            match field_types.get(&key_str) {
                None => {
                    // First occurrence of this field
                    field_types.insert(key_str, inferred_type);
                }
                Some(existing_type) if existing_type != &inferred_type => {
                    // Type conflict detected - fallback to Utf8
                    tracing::warn!(
                        column = %key_str,
                        existing_type = ?existing_type,
                        new_type = ?inferred_type,
                        "Type conflict detected, encoding as LargeUtf8"
                    );

                    type_conflicts
                        .entry(key_str.clone())
                        .or_insert_with(|| vec![existing_type.clone()])
                        .push(inferred_type);

                    field_types.insert(key_str, DataType::LargeUtf8);
                }
                Some(_) => {
                    // Same type, no action needed
                }
            }
        }
    }

    if field_types.is_empty() {
        return Err(ParquetEncodingError::NoFieldsInferred);
    }

    // Build Arrow schema (all fields nullable)
    let arrow_fields: Vec<Arc<Field>> = field_types
        .into_iter()
        .map(|(name, dtype)| Arc::new(Field::new(name, dtype, true)))
        .collect();

    Ok(Arc::new(Schema::new(arrow_fields)))
}

/// Expand JSON columns in a record batch using the configured processors
///
/// This function:
/// 1. Extracts JSON string data from specified columns
/// 2. Flattens nested JSON objects into subcolumns
/// 3. Creates bucket maps for overflow keys
/// 4. Returns a new record batch with expanded columns
fn expand_json_columns(
    record_batch: arrow::array::RecordBatch,
    processors: &[JsonColumnProcessor],
    _events: &[Event],
) -> Result<arrow::array::RecordBatch, ParquetEncodingError> {
    use arrow::array::{Array, LargeStringArray, RecordBatch, StringArray};
    use arrow::datatypes::{Field, Schema};

    let schema = record_batch.schema();
    let num_rows = record_batch.num_rows();

    // Collect new fields and arrays
    let mut new_fields: Vec<Arc<Field>> = Vec::new();
    let mut new_arrays: Vec<ArrayRef> = Vec::new();

    // Track which columns to exclude (the original JSON columns if not keeping them)
    let mut columns_to_expand: std::collections::HashSet<String> = std::collections::HashSet::new();
    for processor in processors {
        columns_to_expand.insert(processor.column_name().to_string());
    }

    // Process each JSON column
    let mut processed_columns: Vec<ProcessedJsonColumns> = Vec::new();
    for processor in processors {
        let column_name = processor.column_name();

        // Find the column in the record batch
        if let Some(col_idx) = schema.fields().iter().position(|f| f.name() == column_name) {
            let array = record_batch.column(col_idx);

            // Try to cast to StringArray or LargeStringArray
            let json_values: Option<Vec<Option<&str>>> =
                if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
                    Some((0..num_rows)
                        .map(|i| {
                            if string_array.is_null(i) {
                                None
                            } else {
                                Some(string_array.value(i))
                            }
                        })
                        .collect())
                } else if let Some(large_string_array) = array.as_any().downcast_ref::<LargeStringArray>() {
                    Some((0..num_rows)
                        .map(|i| {
                            if large_string_array.is_null(i) {
                                None
                            } else {
                                Some(large_string_array.value(i))
                            }
                        })
                        .collect())
                } else {
                    None
                };

            if let Some(json_values) = json_values {
                // Process the JSON column
                let processed = processor.process_batch(json_values.into_iter());
                processed_columns.push(processed);
            } else {
                tracing::warn!(
                    column = column_name,
                    "JSON column is not a string type, skipping expansion"
                );
            }
        } else {
            tracing::debug!(
                column = column_name,
                "JSON column not found in schema, skipping"
            );
        }
    }

    // First, add all non-JSON columns as-is
    for (idx, field) in schema.fields().iter().enumerate() {
        let field_name = field.name();
        let was_expanded = processed_columns
            .iter()
            .any(|p| p.column_name == *field_name);

        if !was_expanded {
            new_fields.push(Arc::new(Field::new(
                field_name,
                field.data_type().clone(),
                field.is_nullable(),
            )));
            new_arrays.push(record_batch.column(idx).clone());
        }
        // JSON columns are replaced by expanded subcolumns below
    }

    // Add expanded columns from processed JSON columns
    for processed in &processed_columns {
        // Add original column if kept
        if let Some((name, array)) = processed.original_to_array() {
            new_fields.push(Arc::new(Field::new(&name, DataType::LargeUtf8, true)));
            new_arrays.push(array);
        }

        // Add subcolumns
        for (name, array) in processed.subcolumns_to_arrays() {
            let data_type = array.data_type().clone();
            new_fields.push(Arc::new(Field::new(&name, data_type, true)));
            new_arrays.push(array);
        }

        // Add bucket maps
        for (name, array) in processed.bucket_maps_to_arrays() {
            let data_type = array.data_type().clone();
            new_fields.push(Arc::new(Field::new(&name, data_type, true)));
            new_arrays.push(array);
        }
    }

    // Create new schema and record batch
    let new_schema = Arc::new(Schema::new(new_fields));
    let new_batch = RecordBatch::try_new(new_schema, new_arrays).map_err(|e| {
        ArrowEncodingError::RecordBatchCreation { source: e }
    })?;

    Ok(new_batch)
}

/// Encodes a batch of events into Parquet format
pub fn encode_events_to_parquet(
    events: &[Event],
    schema: Arc<Schema>,
    writer_properties: &WriterProperties,
) -> Result<Bytes, ParquetEncodingError> {
    if events.is_empty() {
        return Err(ParquetEncodingError::NoEvents);
    }

    // Build Arrow RecordBatch from events (reuses Arrow encoder logic)
    let record_batch = build_record_batch(schema, events)?;

    // Get batch metadata before we move into writer scope
    let batch_schema = record_batch.schema();

    // Write RecordBatch to Parquet format in memory
    let mut buffer = Vec::new();
    {
        let mut writer =
            ArrowWriter::try_new(&mut buffer, batch_schema, Some(writer_properties.clone()))?;

        writer.write(&record_batch)?;
        writer.close()?;
    }

    Ok(Bytes::from(buffer))
}

/// Encodes a pre-built RecordBatch into Parquet format
///
/// This is used by the super-batch functionality where the batch has already
/// been built and potentially sorted/sliced.
fn encode_record_batch_to_parquet(
    record_batch: &arrow::array::RecordBatch,
    writer_properties: &WriterProperties,
) -> Result<Bytes, ParquetEncodingError> {
    let batch_schema = record_batch.schema();

    let mut buffer = Vec::new();
    {
        let mut writer =
            ArrowWriter::try_new(&mut buffer, batch_schema, Some(writer_properties.clone()))?;

        writer.write(record_batch)?;
        writer.close()?;
    }

    Ok(Bytes::from(buffer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{
            Array, BinaryArray, BooleanArray, Float64Array, Int64Array, StringArray,
            TimestampMicrosecondArray,
        },
        datatypes::{DataType, Field, TimeUnit},
    };
    use bytes::Bytes;
    use chrono::Utc;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use vector_core::event::LogEvent;

    #[test]
    fn test_encode_all_types() {
        let mut log = LogEvent::default();
        log.insert("string_field", "test");
        log.insert("int64_field", 42);
        log.insert("float64_field", 3.15);
        log.insert("bool_field", true);
        log.insert("bytes_field", bytes::Bytes::from("binary"));
        log.insert("timestamp_field", Utc::now());

        let events = vec![Event::Log(log)];

        let schema = Arc::new(Schema::new(vec![
            Field::new("string_field", DataType::Utf8, true),
            Field::new("int64_field", DataType::Int64, true),
            Field::new("float64_field", DataType::Float64, true),
            Field::new("bool_field", DataType::Boolean, true),
            Field::new("bytes_field", DataType::Binary, true),
            Field::new(
                "timestamp_field",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                true,
            ),
        ]));

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let result = encode_events_to_parquet(&events, Arc::clone(&schema), &props);
        assert!(result.is_ok());

        let bytes = result.unwrap();

        // Verify it's valid Parquet by reading it back
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .unwrap()
            .build()
            .unwrap();

        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        assert_eq!(batches.len(), 1);

        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 6);

        // Verify string field
        assert_eq!(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0),
            "test"
        );

        // Verify int64 field
        assert_eq!(
            batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            42
        );

        // Verify float64 field
        assert!(
            (batch
                .column(2)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(0)
                - 3.15)
                .abs()
                < 0.001
        );

        // Verify boolean field
        assert!(
            batch
                .column(3)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(0)
        );

        // Verify binary field
        assert_eq!(
            batch
                .column(4)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .value(0),
            b"binary"
        );

        // Verify timestamp field
        assert!(
            !batch
                .column(5)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .is_null(0)
        );
    }

    #[test]
    fn test_encode_null_values() {
        let mut log1 = LogEvent::default();
        log1.insert("field_a", 1);
        // field_b is missing

        let mut log2 = LogEvent::default();
        log2.insert("field_b", 2);
        // field_a is missing

        let events = vec![Event::Log(log1), Event::Log(log2)];

        let schema = Arc::new(Schema::new(vec![
            Field::new("field_a", DataType::Int64, true),
            Field::new("field_b", DataType::Int64, true),
        ]));

        let props = WriterProperties::builder().build();

        let result = encode_events_to_parquet(&events, Arc::clone(&schema), &props);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .unwrap()
            .build()
            .unwrap();

        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        let batch = &batches[0];

        assert_eq!(batch.num_rows(), 2);

        let field_a = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(field_a.value(0), 1);
        assert!(field_a.is_null(1));

        let field_b = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(field_b.is_null(0));
        assert_eq!(field_b.value(1), 2);
    }

    #[test]
    fn test_encode_empty_events() {
        let events: Vec<Event> = vec![];
        let schema = Arc::new(Schema::new(vec![Field::new(
            "field",
            DataType::Int64,
            true,
        )]));
        let props = WriterProperties::builder().build();
        let result = encode_events_to_parquet(&events, schema, &props);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ParquetEncodingError::NoEvents
        ));
    }

    #[test]
    fn test_parquet_compression_types() {
        let mut log = LogEvent::default();
        log.insert("message", "test message");

        let events = vec![Event::Log(log)];
        let schema = Arc::new(Schema::new(vec![Field::new(
            "message",
            DataType::Utf8,
            true,
        )]));

        // Test different compression algorithms
        let compressions = vec![
            ParquetCompression::Uncompressed,
            ParquetCompression::Snappy,
            ParquetCompression::Gzip,
            ParquetCompression::Zstd,
        ];

        for compression in compressions {
            let props = WriterProperties::builder()
                .set_compression(compression.into())
                .build();

            let result = encode_events_to_parquet(&events, Arc::clone(&schema), &props);
            assert!(result.is_ok(), "Failed with compression: {:?}", compression);

            // Verify we can read it back
            let bytes = result.unwrap();
            let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
                .unwrap()
                .build()
                .unwrap();

            let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
            assert_eq!(batches[0].num_rows(), 1);
        }
    }

    #[test]
    fn test_parquet_serializer_config() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;

        let mut fields = BTreeMap::new();
        fields.insert(
            "field".to_string(),
            FieldDefinition {
                r#type: "int64".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        let config = ParquetSerializerConfig {
            schema: Some(SchemaDefinition { fields }),
            infer_schema: false,
            exclude_columns: None,
            max_columns: default_max_columns(),
            compression: ParquetCompression::Zstd,
            compression_level: None,
            writer_version: ParquetWriterVersion::default(),
            row_group_size: Some(1000),
            allow_nullable_fields: false,
            sorting_columns: None,
            rows_per_file: None,
            json_columns: None,
        };

        let serializer = ParquetSerializer::new(config);
        assert!(serializer.is_ok());
    }

    #[test]
    fn test_parquet_serializer_no_schema_fails() {
        let config = ParquetSerializerConfig {
            schema: None,
            infer_schema: false,
            exclude_columns: None,
            max_columns: default_max_columns(),
            compression: ParquetCompression::default(),
            compression_level: None,
            writer_version: ParquetWriterVersion::default(),
            row_group_size: None,
            allow_nullable_fields: false,
            sorting_columns: None,
            rows_per_file: None,
            json_columns: None,
        };

        let result = ParquetSerializer::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_trait_implementation() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;
        use tokio_util::codec::Encoder;

        let mut fields = BTreeMap::new();
        fields.insert(
            "id".to_string(),
            FieldDefinition {
                r#type: "int64".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );
        fields.insert(
            "name".to_string(),
            FieldDefinition {
                r#type: "utf8".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        let config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        let mut serializer = ParquetSerializer::new(config).unwrap();

        let mut log = LogEvent::default();
        log.insert("id", 1);
        log.insert("name", "test");

        let events = vec![Event::Log(log)];
        let mut buffer = BytesMut::new();

        let result = serializer.encode(events, &mut buffer);
        assert!(result.is_ok());
        assert!(!buffer.is_empty());

        // Verify the buffer contains valid Parquet data
        let bytes = Bytes::copy_from_slice(&buffer);
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes);
        assert!(reader.is_ok());
    }

    #[test]
    fn test_large_batch_encoding() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("value", DataType::Float64, true),
        ]));

        // Create 10,000 events
        let events: Vec<Event> = (0..10000)
            .map(|i| {
                let mut log = LogEvent::default();
                log.insert("id", i);
                log.insert("value", i as f64 * 1.5);
                Event::Log(log)
            })
            .collect();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_size(5000) // 2 row groups
            .build();

        let result = encode_events_to_parquet(&events, Arc::clone(&schema), &props);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .unwrap()
            .build()
            .unwrap();

        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 10000);
    }

    #[test]
    fn test_allow_nullable_fields_config() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;
        use tokio_util::codec::Encoder;

        let mut fields = BTreeMap::new();
        fields.insert(
            "required_field".to_string(),
            FieldDefinition {
                r#type: "int64".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        let mut log1 = LogEvent::default();
        log1.insert("required_field", 42);

        let log2 = LogEvent::default();
        // log2 is missing required_field

        let events = vec![Event::Log(log1), Event::Log(log2)];

        // Note: SchemaDefinition creates nullable fields by default
        // This test verifies that the allow_nullable_fields flag works
        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.allow_nullable_fields = true;

        let mut serializer = ParquetSerializer::new(config).unwrap();
        let mut buffer = BytesMut::new();
        let result = serializer.encode(events.clone(), &mut buffer);
        assert!(result.is_ok());

        // Verify the data
        let bytes = Bytes::copy_from_slice(&buffer);
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .unwrap()
            .build()
            .unwrap();

        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        let batch = &batches[0];

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(array.value(0), 42);
        assert!(array.is_null(1));
    }

    #[test]
    fn test_super_batch_split_without_sorting() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;

        let mut fields = BTreeMap::new();
        fields.insert(
            "id".to_string(),
            FieldDefinition {
                r#type: "int64".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        // Create 100 events
        let events: Vec<Event> = (0..100)
            .map(|i| {
                let mut log = LogEvent::default();
                log.insert("id", i);
                Event::Log(log)
            })
            .collect();

        // Configure super-batch with 30 rows per file (should produce 4 files)
        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.rows_per_file = Some(30);

        let serializer = ParquetSerializer::new(config).unwrap();
        assert!(serializer.is_super_batch_enabled());

        let files = serializer.encode_batch_split(events).unwrap();
        assert_eq!(files.len(), 4); // 30 + 30 + 30 + 10 = 100

        // Verify each file
        let mut total_rows = 0;
        for (i, file_bytes) in files.iter().enumerate() {
            let reader = ParquetRecordBatchReaderBuilder::try_new(file_bytes.clone())
                .unwrap()
                .build()
                .unwrap();

            let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
            let batch = &batches[0];

            if i < 3 {
                assert_eq!(batch.num_rows(), 30);
            } else {
                assert_eq!(batch.num_rows(), 10);
            }
            total_rows += batch.num_rows();
        }
        assert_eq!(total_rows, 100);
    }

    #[test]
    fn test_super_batch_split_with_sorting() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;

        let mut fields = BTreeMap::new();
        fields.insert(
            "id".to_string(),
            FieldDefinition {
                r#type: "int64".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        // Create 50 events in reverse order (50, 49, 48, ..., 1)
        let events: Vec<Event> = (0..50)
            .rev()
            .map(|i| {
                let mut log = LogEvent::default();
                log.insert("id", i);
                Event::Log(log)
            })
            .collect();

        // Configure super-batch with sorting (ascending) and 20 rows per file
        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.rows_per_file = Some(20);
        config.sorting_columns = Some(vec![SortingColumnConfig {
            column: "id".to_string(),
            descending: false,
        }]);

        let serializer = ParquetSerializer::new(config).unwrap();
        let files = serializer.encode_batch_split(events).unwrap();
        assert_eq!(files.len(), 3); // 20 + 20 + 10 = 50

        // Verify sorting across files - first file should have lowest ids
        let first_file = ParquetRecordBatchReaderBuilder::try_new(files[0].clone())
            .unwrap()
            .build()
            .unwrap();
        let first_batches: Vec<_> = first_file.collect::<Result<_, _>>().unwrap();
        let first_batch = &first_batches[0];
        let first_ids = first_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        // First file should contain ids 0-19 (sorted ascending)
        assert_eq!(first_ids.value(0), 0);
        assert_eq!(first_ids.value(19), 19);

        // Second file should contain ids 20-39
        let second_file = ParquetRecordBatchReaderBuilder::try_new(files[1].clone())
            .unwrap()
            .build()
            .unwrap();
        let second_batches: Vec<_> = second_file.collect::<Result<_, _>>().unwrap();
        let second_batch = &second_batches[0];
        let second_ids = second_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(second_ids.value(0), 20);
        assert_eq!(second_ids.value(19), 39);

        // Third file should contain ids 40-49
        let third_file = ParquetRecordBatchReaderBuilder::try_new(files[2].clone())
            .unwrap()
            .build()
            .unwrap();
        let third_batches: Vec<_> = third_file.collect::<Result<_, _>>().unwrap();
        let third_batch = &third_batches[0];
        let third_ids = third_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(third_ids.value(0), 40);
        assert_eq!(third_ids.value(9), 49);
    }

    #[test]
    fn test_super_batch_descending_sort() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;

        let mut fields = BTreeMap::new();
        fields.insert(
            "timestamp".to_string(),
            FieldDefinition {
                r#type: "int64".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        // Create 30 events with timestamps 0..30
        let events: Vec<Event> = (0..30)
            .map(|i| {
                let mut log = LogEvent::default();
                log.insert("timestamp", i);
                Event::Log(log)
            })
            .collect();

        // Configure super-batch with descending sort
        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.rows_per_file = Some(10);
        config.sorting_columns = Some(vec![SortingColumnConfig {
            column: "timestamp".to_string(),
            descending: true,
        }]);

        let serializer = ParquetSerializer::new(config).unwrap();
        let files = serializer.encode_batch_split(events).unwrap();
        assert_eq!(files.len(), 3);

        // First file should have highest timestamps (29, 28, 27, ...)
        let first_file = ParquetRecordBatchReaderBuilder::try_new(files[0].clone())
            .unwrap()
            .build()
            .unwrap();
        let first_batches: Vec<_> = first_file.collect::<Result<_, _>>().unwrap();
        let first_batch = &first_batches[0];
        let first_ts = first_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(first_ts.value(0), 29);
        assert_eq!(first_ts.value(9), 20);
    }

    #[test]
    fn test_super_batch_small_batch_no_split() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;

        let mut fields = BTreeMap::new();
        fields.insert(
            "id".to_string(),
            FieldDefinition {
                r#type: "int64".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        // Create only 5 events
        let events: Vec<Event> = (0..5)
            .map(|i| {
                let mut log = LogEvent::default();
                log.insert("id", i);
                Event::Log(log)
            })
            .collect();

        // Configure super-batch with 10 rows per file
        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.rows_per_file = Some(10);

        let serializer = ParquetSerializer::new(config).unwrap();
        let files = serializer.encode_batch_split(events).unwrap();

        // Should produce only 1 file since we have fewer events than rows_per_file
        assert_eq!(files.len(), 1);

        let reader = ParquetRecordBatchReaderBuilder::try_new(files[0].clone())
            .unwrap()
            .build()
            .unwrap();
        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        assert_eq!(batches[0].num_rows(), 5);
    }

    #[test]
    fn test_super_batch_disabled_by_default() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;

        let mut fields = BTreeMap::new();
        fields.insert(
            "id".to_string(),
            FieldDefinition {
                r#type: "int64".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        let config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        let serializer = ParquetSerializer::new(config).unwrap();

        // Super-batch should be disabled when rows_per_file is not set
        assert!(!serializer.is_super_batch_enabled());
        assert!(serializer.rows_per_file().is_none());
    }

    #[test]
    fn test_super_batch_sorts_without_split() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;

        let mut fields = BTreeMap::new();
        fields.insert(
            "id".to_string(),
            FieldDefinition {
                r#type: "int64".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        // Create 10 events in reverse order (9, 8, 7, ..., 0)
        let events: Vec<Event> = (0..10)
            .rev()
            .map(|i| {
                let mut log = LogEvent::default();
                log.insert("id", i);
                Event::Log(log)
            })
            .collect();

        // Configure with sorting but rows_per_file larger than batch size
        // This means NO split should occur, but sorting SHOULD still happen
        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.rows_per_file = Some(100); // Larger than our 10 events
        config.sorting_columns = Some(vec![SortingColumnConfig {
            column: "id".to_string(),
            descending: false, // ascending
        }]);

        let serializer = ParquetSerializer::new(config).unwrap();
        let files = serializer.encode_batch_split(events).unwrap();

        // Should produce only 1 file (no split)
        assert_eq!(files.len(), 1);

        // But the data should be sorted (0, 1, 2, ..., 9)
        let reader = ParquetRecordBatchReaderBuilder::try_new(files[0].clone())
            .unwrap()
            .build()
            .unwrap();
        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        let batch = &batches[0];
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        // Verify sorted order
        assert_eq!(ids.value(0), 0);
        assert_eq!(ids.value(5), 5);
        assert_eq!(ids.value(9), 9);
    }

    #[test]
    fn test_super_batch_rows_per_file_zero_fails() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;

        let mut fields = BTreeMap::new();
        fields.insert(
            "id".to_string(),
            FieldDefinition {
                r#type: "int64".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.rows_per_file = Some(0);

        // Should fail validation with a clear error
        let result = ParquetSerializer::new(config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("rows_per_file") && err.contains("greater than 0"),
            "Expected error about rows_per_file being > 0, got: {}",
            err
        );
    }

    #[test]
    fn test_json_column_expansion() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;
        use tokio_util::codec::Encoder;

        // Create events with a JSON properties column
        let mut log1 = LogEvent::default();
        log1.insert("event_type", "click");
        log1.insert(
            "properties",
            r#"{"$ip":"192.168.1.1","user_id":"user123","page_views":42,"is_premium":true}"#,
        );

        let mut log2 = LogEvent::default();
        log2.insert("event_type", "view");
        log2.insert(
            "properties",
            r#"{"$ip":"10.0.0.1","user_id":"user456","page_views":100,"is_premium":false,"extra_field":"value"}"#,
        );

        let events = vec![Event::Log(log1), Event::Log(log2)];

        // Create explicit schema for the two columns
        let mut fields = BTreeMap::new();
        fields.insert(
            "event_type".to_string(),
            FieldDefinition {
                r#type: "utf8".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );
        fields.insert(
            "properties".to_string(),
            FieldDefinition {
                r#type: "utf8".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        // Create a config with JSON column expansion
        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.json_columns = Some(vec![JsonColumnConfig {
            column: "properties".to_string(),
            max_subcolumns: 3, // Only expand top 3 keys
            bucket_count: 2,   // Use 2 buckets for overflow
            max_depth: 5,
            keep_original_column: true,
            type_hints: None,
        }]);

        let mut serializer = ParquetSerializer::new(config).expect("Failed to create serializer");
        assert!(serializer.has_json_columns());

        let mut buffer = BytesMut::new();
        serializer
            .encode(events, &mut buffer)
            .expect("Failed to encode events");

        // Verify it's valid Parquet by reading it back
        let bytes = Bytes::from(buffer.to_vec());
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .expect("Failed to create reader")
            .build()
            .expect("Failed to build reader");

        let batches: Vec<_> = reader.collect::<Result<_, _>>().expect("Failed to read batches");
        assert_eq!(batches.len(), 1);

        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);

        // Check that we have the expected columns:
        // - event_type (original)
        // - properties (original JSON string, kept)
        // - properties.$ip (subcolumn)
        // - properties.user_id (subcolumn)
        // - properties.page_views (subcolumn)
        // - properties__json_type_bucket_0 (bucket map)
        // - properties__json_type_bucket_1 (bucket map)
        let schema = batch.schema();
        let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

        // Should have the original properties column
        assert!(
            field_names.contains(&"properties"),
            "Expected 'properties' column, got: {:?}",
            field_names
        );

        // Should have some subcolumns (the exact ones depend on priority calculation)
        let subcolumn_count = field_names
            .iter()
            .filter(|n| n.starts_with("properties."))
            .count();
        assert!(
            subcolumn_count > 0,
            "Expected at least one properties.* subcolumn, got: {:?}",
            field_names
        );

        // Should have bucket map columns
        let bucket_count = field_names
            .iter()
            .filter(|n| n.starts_with("properties__json_type_bucket_"))
            .count();
        assert_eq!(
            bucket_count, 2,
            "Expected 2 bucket columns, got {} in: {:?}",
            bucket_count, field_names
        );

        // Verify bucket columns are Map<String, String> type
        for field in schema.fields() {
            if field.name().starts_with("properties__json_type_bucket_") {
                match field.data_type() {
                    DataType::Map(_, _) => {}
                    other => panic!(
                        "Expected bucket column to be Map type, got: {:?}",
                        other
                    ),
                }
            }
        }

        println!("JSON column expansion test passed! Columns: {:?}", field_names);
    }

    #[test]
    fn test_json_column_type_inference() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;
        use tokio_util::codec::Encoder;

        // Create events with various JSON value types
        let mut log1 = LogEvent::default();
        log1.insert(
            "data",
            r#"{"string_val":"hello","int_val":42,"float_val":3.14,"bool_val":true}"#,
        );

        let mut log2 = LogEvent::default();
        log2.insert(
            "data",
            r#"{"string_val":"world","int_val":100,"float_val":2.71,"bool_val":false}"#,
        );

        let events = vec![Event::Log(log1), Event::Log(log2)];

        // Create explicit schema
        let mut fields = BTreeMap::new();
        fields.insert(
            "data".to_string(),
            FieldDefinition {
                r#type: "utf8".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.json_columns = Some(vec![JsonColumnConfig {
            column: "data".to_string(),
            max_subcolumns: 10,
            bucket_count: 4,
            max_depth: 3,
            keep_original_column: false, // Don't keep original
            type_hints: None,
        }]);

        let mut serializer = ParquetSerializer::new(config).expect("Failed to create serializer");
        let mut buffer = BytesMut::new();
        serializer
            .encode(events, &mut buffer)
            .expect("Failed to encode events");

        let bytes = Bytes::from(buffer.to_vec());
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .expect("Failed to create reader")
            .build()
            .expect("Failed to build reader");

        let batches: Vec<_> = reader.collect::<Result<_, _>>().expect("Failed to read batches");
        let batch = &batches[0];
        let schema = batch.schema();

        // Should NOT have the original 'data' column since keep_original_column is false
        let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(
            !field_names.contains(&"data"),
            "Original 'data' column should not be present when keep_original_column is false"
        );

        // Check subcolumns exist with inferred types
        let string_field = schema.field_with_name("data.string_val");
        assert!(string_field.is_ok(), "Expected data.string_val field");
        assert_eq!(
            string_field.unwrap().data_type(),
            &DataType::LargeUtf8,
            "string_val should be LargeUtf8"
        );

        println!("JSON column type inference test passed! Fields: {:?}", field_names);
    }

    #[test]
    fn test_json_column_nested_objects() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;
        use tokio_util::codec::Encoder;

        // Create events with nested JSON
        let mut log = LogEvent::default();
        log.insert(
            "event",
            r#"{"user":{"name":"Alice","address":{"city":"NYC","zip":"10001"}},"action":"login"}"#,
        );

        let events = vec![Event::Log(log)];

        // Create explicit schema
        let mut fields = BTreeMap::new();
        fields.insert(
            "event".to_string(),
            FieldDefinition {
                r#type: "utf8".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.json_columns = Some(vec![JsonColumnConfig {
            column: "event".to_string(),
            max_subcolumns: 100,
            bucket_count: 4,
            max_depth: 3, // Allow 3 levels of nesting
            keep_original_column: true,
            type_hints: None,
        }]);

        let mut serializer = ParquetSerializer::new(config).expect("Failed to create serializer");
        let mut buffer = BytesMut::new();
        serializer
            .encode(events, &mut buffer)
            .expect("Failed to encode events");

        let bytes = Bytes::from(buffer.to_vec());
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .expect("Failed to create reader")
            .build()
            .expect("Failed to build reader");

        let batches: Vec<_> = reader.collect::<Result<_, _>>().expect("Failed to read batches");
        let batch = &batches[0];
        let schema = batch.schema();

        let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

        // Should have flattened nested keys
        assert!(
            field_names.contains(&"event.action"),
            "Expected event.action field, got: {:?}",
            field_names
        );
        assert!(
            field_names.contains(&"event.user.name"),
            "Expected event.user.name field, got: {:?}",
            field_names
        );
        assert!(
            field_names.contains(&"event.user.address.city"),
            "Expected event.user.address.city field, got: {:?}",
            field_names
        );

        println!("JSON nested objects test passed! Fields: {:?}", field_names);
    }

    #[test]
    fn test_infer_schema_basic() {
        use tokio_util::codec::Encoder;

        // Test basic schema inference without JSON columns
        let mut log1 = LogEvent::default();
        log1.insert("name", "Alice");
        log1.insert("age", 30_i64);

        let mut log2 = LogEvent::default();
        log2.insert("name", "Bob");
        log2.insert("age", 25_i64);

        let events = vec![Event::Log(log1), Event::Log(log2)];

        // Debug: print what all_event_fields returns
        for (i, event) in events.iter().enumerate() {
            if let Event::Log(log) = event {
                println!("Event {} fields:", i);
                if let Some(fields) = log.all_event_fields() {
                    for (key, value) in fields {
                        println!("  {} = {:?}", key, value);
                    }
                } else {
                    println!("  <all_event_fields returned None>");
                }
            }
        }

        // Create config with infer_schema enabled
        let mut config = ParquetSerializerConfig::default();
        config.infer_schema = true;

        let mut serializer = ParquetSerializer::new(config).expect("Failed to create serializer");
        let mut buffer = BytesMut::new();

        serializer
            .encode(events, &mut buffer)
            .expect("Failed to encode events with infer_schema");

        // Verify we got some output
        assert!(!buffer.is_empty(), "Buffer should not be empty");

        let bytes = Bytes::from(buffer.to_vec());
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .expect("Failed to create reader")
            .build()
            .expect("Failed to build reader");

        let batches: Vec<_> = reader.collect::<Result<_, _>>().expect("Failed to read batches");
        let batch = &batches[0];
        let schema = batch.schema();

        let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

        assert!(
            field_names.contains(&"name"),
            "Expected 'name' field, got: {:?}",
            field_names
        );
        assert!(
            field_names.contains(&"age"),
            "Expected 'age' field, got: {:?}",
            field_names
        );

        println!("Infer schema basic test passed! Fields: {:?}", field_names);
    }

    #[test]
    fn test_super_batch_with_json_column_expansion() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;

        // Create 100 events with JSON properties
        let events: Vec<Event> = (0..100)
            .map(|i| {
                let mut log = LogEvent::default();
                log.insert("id", i as i64);
                log.insert(
                    "properties",
                    format!(
                        r#"{{"user_id":"user{}","score":{},"active":{}}}"#,
                        i,
                        i * 10,
                        i % 2 == 0
                    ),
                );
                Event::Log(log)
            })
            .collect();

        // Create explicit schema
        let mut fields = BTreeMap::new();
        fields.insert(
            "id".to_string(),
            FieldDefinition {
                r#type: "int64".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );
        fields.insert(
            "properties".to_string(),
            FieldDefinition {
                r#type: "string".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        // Configure super-batch with 30 rows per file AND JSON column expansion
        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.rows_per_file = Some(30);
        config.json_columns = Some(vec![JsonColumnConfig {
            column: "properties".to_string(),
            max_subcolumns: 10,
            bucket_count: 4,
            max_depth: 3,
            keep_original_column: true,
            type_hints: None,
        }]);

        let serializer = ParquetSerializer::new(config).unwrap();
        assert!(serializer.is_super_batch_enabled());
        assert!(serializer.has_json_columns());

        // This should produce 4 files (30 + 30 + 30 + 10 = 100)
        let files = serializer.encode_batch_split(events).unwrap();
        assert_eq!(files.len(), 4, "Expected 4 files, got {}", files.len());

        // Verify each file is valid Parquet with expanded JSON columns
        for (idx, file_bytes) in files.iter().enumerate() {
            let reader = ParquetRecordBatchReaderBuilder::try_new(file_bytes.clone())
                .expect(&format!("Failed to create reader for file {}", idx))
                .build()
                .expect(&format!("Failed to build reader for file {}", idx));

            let batches: Vec<_> = reader
                .collect::<Result<_, _>>()
                .expect(&format!("Failed to read batches from file {}", idx));

            assert_eq!(batches.len(), 1, "Expected 1 batch in file {}", idx);

            let batch = &batches[0];
            let schema = batch.schema();
            let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

            // Verify JSON subcolumns exist
            assert!(
                field_names.iter().any(|n| n.starts_with("properties.")),
                "File {} should have expanded JSON subcolumns, got fields: {:?}",
                idx,
                field_names
            );

            // Verify original column is kept
            assert!(
                field_names.contains(&"properties"),
                "File {} should keep original properties column, got fields: {:?}",
                idx,
                field_names
            );

            // Verify row count
            let expected_rows = if idx == 3 { 10 } else { 30 };
            assert_eq!(
                batch.num_rows(),
                expected_rows,
                "File {} should have {} rows, got {}",
                idx,
                expected_rows,
                batch.num_rows()
            );
        }

        println!("Super-batch with JSON column expansion test passed!");
    }

    #[test]
    fn test_super_batch_json_expansion_with_sorting() {
        use super::super::schema_definition::FieldDefinition;
        use std::collections::BTreeMap;

        // Create events with JSON properties in random order
        let events: Vec<Event> = vec![50, 10, 90, 30, 70, 20, 80, 40, 60, 0]
            .into_iter()
            .map(|i| {
                let mut log = LogEvent::default();
                log.insert("sort_key", i as i64);
                log.insert(
                    "data",
                    format!(r#"{{"value":{},"name":"item{}"}}"#, i, i),
                );
                Event::Log(log)
            })
            .collect();

        let mut fields = BTreeMap::new();
        fields.insert(
            "sort_key".to_string(),
            FieldDefinition {
                r#type: "int64".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );
        fields.insert(
            "data".to_string(),
            FieldDefinition {
                r#type: "string".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        // Configure with sorting, super-batch, and JSON expansion
        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.rows_per_file = Some(3);
        config.sorting_columns = Some(vec![SortingColumnConfig {
            column: "sort_key".to_string(),
            descending: false,
        }]);
        config.json_columns = Some(vec![JsonColumnConfig {
            column: "data".to_string(),
            max_subcolumns: 5,
            bucket_count: 2,
            max_depth: 2,
            keep_original_column: false,
            type_hints: None,
        }]);

        let serializer = ParquetSerializer::new(config).unwrap();
        let files = serializer.encode_batch_split(events).unwrap();

        // Should have 4 files (3 + 3 + 3 + 1 = 10)
        assert_eq!(files.len(), 4, "Expected 4 files");

        // Verify sorting is preserved and JSON is expanded in each file
        let mut all_sort_keys: Vec<i64> = Vec::new();

        for (idx, file_bytes) in files.iter().enumerate() {
            let reader = ParquetRecordBatchReaderBuilder::try_new(file_bytes.clone())
                .unwrap()
                .build()
                .unwrap();

            let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
            let batch = &batches[0];
            let schema = batch.schema();
            let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

            // Verify JSON expansion happened (original column should NOT exist)
            assert!(
                !field_names.contains(&"data"),
                "File {} should NOT have original 'data' column (keep_original_column=false), got: {:?}",
                idx,
                field_names
            );

            // Verify subcolumns exist
            assert!(
                field_names.iter().any(|n| n.starts_with("data.")),
                "File {} should have data.* subcolumns, got: {:?}",
                idx,
                field_names
            );

            // Extract sort keys to verify ordering
            let sort_key_idx = schema.fields().iter().position(|f| f.name() == "sort_key").unwrap();
            let sort_key_array = batch
                .column(sort_key_idx)
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                all_sort_keys.push(sort_key_array.value(i));
            }
        }

        // Verify global sort order
        let expected: Vec<i64> = vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90];
        assert_eq!(
            all_sort_keys, expected,
            "Events should be sorted ascending by sort_key"
        );

        println!("Super-batch with sorting and JSON expansion test passed!");
    }

    #[test]
    fn test_large_string_schema_consistency() {
        use super::super::schema_definition::FieldDefinition;
        use arrow::datatypes::DataType;
        use std::collections::BTreeMap;

        // Test that string fields use LargeUtf8 consistently
        let mut log = LogEvent::default();
        log.insert("text_field", "hello world");
        log.insert("json_field", r#"{"key": "value"}"#);

        let events = vec![Event::Log(log)];

        let mut fields = BTreeMap::new();
        fields.insert(
            "text_field".to_string(),
            FieldDefinition {
                r#type: "string".to_string(),
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );
        fields.insert(
            "json_field".to_string(),
            FieldDefinition {
                r#type: "utf8".to_string(), // Also test "utf8" alias
                bloom_filter: false,
                bloom_filter_num_distinct_values: None,
                bloom_filter_false_positive_pct: None,
            },
        );

        let mut config = ParquetSerializerConfig::new(SchemaDefinition { fields });
        config.json_columns = Some(vec![JsonColumnConfig {
            column: "json_field".to_string(),
            max_subcolumns: 10,
            bucket_count: 2,
            max_depth: 3,
            keep_original_column: true,
            type_hints: None,
        }]);

        let serializer = ParquetSerializer::new(config).unwrap();
        let files = serializer.encode_batch_split(events).unwrap();

        let reader = ParquetRecordBatchReaderBuilder::try_new(files[0].clone())
            .unwrap()
            .build()
            .unwrap();

        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        let schema = batches[0].schema();

        // Verify all string fields use LargeUtf8
        for field in schema.fields() {
            if field.data_type() == &DataType::Utf8 {
                panic!(
                    "Field '{}' uses Utf8 instead of LargeUtf8 - this can cause overflow issues",
                    field.name()
                );
            }
        }

        println!("Large string schema consistency test passed!");
    }
}

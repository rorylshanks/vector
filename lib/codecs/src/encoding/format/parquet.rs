use arrow::{
    array::{
        ArrayRef, BinaryBuilder, BooleanBuilder, Decimal128Builder, Float32Builder, Float64Builder,
        Int8Builder, Int16Builder, Int32Builder, Int64Builder, ListBuilder, StringBuilder,
    },
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use bytes::{BufMut, BytesMut};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, Encoding},
    file::properties::WriterProperties,
};
use rust_decimal::Decimal;
use snafu::Snafu;
use std::sync::Arc;
use tokio_util::codec::Encoder;
use vector_config::configurable_component;
use vector_core::{
    config::DataType as VectorDataType,
    event::{Event, Value},
    schema,
};

use crate::encoding::BuildError;

/// Config used to build a `ParquetSerializer`.
#[configurable_component]
#[derive(Debug, Clone)]
pub struct ParquetSerializerConfig {
    /// Options for the Parquet serializer.
    pub parquet: ParquetSerializerOptions,
}

impl ParquetSerializerConfig {
    /// Build the `ParquetSerializer` from this configuration.
    pub fn build(&self) -> Result<ParquetSerializer, BuildError> {
        let schema = build_arrow_schema(&self.parquet.schema)?;
        let writer_properties = WriterProperties::builder()
            .set_compression(self.parquet.default_compression.into())
            .set_encoding(self.parquet.default_encoding.into())
            .build();

        Ok(ParquetSerializer {
            schema: Arc::new(schema),
            writer_properties: Arc::new(writer_properties),
        })
    }

    /// The data type of events that are accepted by `ParquetSerializer`.
    pub fn input_type(&self) -> VectorDataType {
        VectorDataType::Log
    }

    /// The schema required by the serializer.
    pub fn schema_requirement(&self) -> schema::Requirement {
        // Schema validation is handled internally against the configured Parquet schema.
        schema::Requirement::empty()
    }
}

/// Parquet serializer options.
#[configurable_component]
#[derive(Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct ParquetSerializerOptions {
    /// Field definitions that make up the Parquet schema.
    #[serde(default)]
    pub schema: Vec<ParquetField>,

    /// The default compression type to use for fields.
    #[serde(default)]
    pub default_compression: ParquetCompression,

    /// The default encoding type to use for fields.
    #[serde(default)]
    pub default_encoding: ParquetEncoding,
}

/// Schema field definition for Parquet output.
#[configurable_component]
#[derive(Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct ParquetField {
    /// The name of the column.
    pub name: String,

    /// The Parquet type for the column.
    #[serde(rename = "type")]
    pub r#type: ParquetFieldType,

    /// Precision to use for DECIMAL32/DECIMAL64 type.
    #[serde(default)]
    pub decimal_precision: u8,

    /// Scale to use for DECIMAL32/DECIMAL64 type.
    #[serde(default)]
    pub decimal_scale: i8,

    /// Whether the field is repeated.
    #[serde(default)]
    pub repeated: bool,

    /// Whether the field is optional.
    #[serde(default)]
    pub optional: bool,

    /// A list of child fields.
    #[serde(default)]
    pub fields: Vec<ParquetField>,
}

/// Supported Parquet leaf and container types.
#[configurable_component]
#[derive(Debug, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ParquetFieldType {
    /// Boolean values.
    Boolean,
    /// 8-bit signed integer.
    Int8,
    /// 16-bit signed integer.
    Int16,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit decimal stored as 128-bit.
    Decimal32,
    /// 64-bit decimal stored as 128-bit.
    Decimal64,
    /// 32-bit floating point.
    Float,
    /// 64-bit floating point.
    Double,
    /// Binary data.
    ByteArray,
    /// UTF-8 strings.
    Utf8,
    /// Map type (string keys).
    Map,
    /// List type.
    List,
}

/// Compression codec to use for Parquet output.
#[configurable_component]
#[derive(Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ParquetCompression {
    /// No compression.
    #[serde(rename = "uncompressed")]
    Uncompressed,
    /// Snappy compression.
    Snappy,
    /// Gzip compression.
    Gzip,
    /// Brotli compression.
    Brotli,
    /// Zstd compression.
    Zstd,
    /// Lz4 raw compression.
    #[serde(rename = "lz4raw")]
    Lz4Raw,
}

impl Default for ParquetCompression {
    fn default() -> Self {
        ParquetCompression::Uncompressed
    }
}

impl From<ParquetCompression> for Compression {
    fn from(compression: ParquetCompression) -> Self {
        match compression {
            ParquetCompression::Uncompressed => Compression::UNCOMPRESSED,
            ParquetCompression::Snappy => Compression::SNAPPY,
            ParquetCompression::Gzip => Compression::GZIP(parquet::basic::GzipLevel::default()),
            ParquetCompression::Brotli => {
                Compression::BROTLI(parquet::basic::BrotliLevel::default())
            }
            ParquetCompression::Zstd => Compression::ZSTD(parquet::basic::ZstdLevel::default()),
            ParquetCompression::Lz4Raw => Compression::LZ4_RAW,
        }
    }
}

/// Encoding to use for Parquet fields.
#[configurable_component]
#[derive(Debug, Clone, Copy)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ParquetEncoding {
    /// Delta length byte array encoding.
    DeltaLengthByteArray,
    /// Plain encoding.
    Plain,
}

impl Default for ParquetEncoding {
    fn default() -> Self {
        ParquetEncoding::DeltaLengthByteArray
    }
}

impl From<ParquetEncoding> for Encoding {
    fn from(encoding: ParquetEncoding) -> Self {
        match encoding {
            ParquetEncoding::DeltaLengthByteArray => Encoding::DELTA_LENGTH_BYTE_ARRAY,
            ParquetEncoding::Plain => Encoding::PLAIN,
        }
    }
}

/// Errors produced while building or encoding Parquet data.
#[derive(Debug, Snafu)]
pub enum ParquetError {
    /// Parquet schema must contain at least one field.
    #[snafu(display("Parquet schema must contain at least one field"))]
    EmptySchema,

    /// Decimal precision must be specified for decimal fields.
    #[snafu(display("Decimal precision must be greater than zero for field '{field_name}'"))]
    DecimalPrecisionRequired { field_name: String },

    /// Decimal precision/scale validation failed.
    #[snafu(display(
        "Invalid decimal precision/scale for field '{field_name}': precision={}, scale={}",
        precision,
        scale
    ))]
    InvalidDecimal {
        field_name: String,
        precision: u8,
        scale: i8,
    },

    /// Map types are not supported yet.
    #[snafu(display("Map type is not supported for field '{field_name}'"))]
    UnsupportedMap { field_name: String },

    /// List definitions require an element field.
    #[snafu(display("List field '{field_name}' requires exactly one child element definition"))]
    MissingListChild { field_name: String },

    /// Null value encountered for a non-nullable field.
    #[snafu(display("Null value for non-nullable field '{field_name}'"))]
    NullConstraint { field_name: String },

    /// Unsupported Arrow data type for the Parquet serializer.
    #[snafu(display("Unsupported Arrow type {:?} for field '{}'", data_type, field_name))]
    UnsupportedType {
        field_name: String,
        data_type: DataType,
    },

    /// Failed while building Arrow arrays or record batches.
    #[snafu(display("Failed to build Parquet record batch: {}", source))]
    ArrowError {
        /// Underlying Arrow error.
        source: arrow::error::ArrowError,
    },

    /// Failed while writing Parquet data.
    #[snafu(display("Failed to write Parquet data: {}", source))]
    WriteError {
        /// Underlying Parquet writer error.
        source: parquet::errors::ParquetError,
    },
}

/// Serializer that converts an `Event` to bytes using the Parquet format.
#[derive(Debug, Clone)]
pub struct ParquetSerializer {
    schema: Arc<Schema>,
    writer_properties: Arc<WriterProperties>,
}

impl ParquetSerializer {
    pub(crate) fn encode_events(
        &self,
        events: &[Event],
        buffer: &mut BytesMut,
    ) -> Result<(), ParquetError> {
        if events.is_empty() {
            return Ok(());
        }

        let batch = build_record_batch(self.schema.clone(), events)?;
        let writer_props = self.writer_properties.as_ref().clone();

        {
            let mut writer =
                ArrowWriter::try_new(buffer.writer(), self.schema.clone(), Some(writer_props))
                    .map_err(|source| ParquetError::WriteError { source })?;
            writer
                .write(&batch)
                .map_err(|source| ParquetError::WriteError { source })?;
            writer
                .close()
                .map_err(|source| ParquetError::WriteError { source })?;
        }

        Ok(())
    }
}

impl Encoder<Event> for ParquetSerializer {
    type Error = vector_common::Error;

    fn encode(&mut self, event: Event, buffer: &mut BytesMut) -> Result<(), Self::Error> {
        self.encode_events(&[event], buffer)
            .map_err(|err| Box::new(err) as _)
    }
}

fn build_arrow_schema(fields: &[ParquetField]) -> Result<Schema, ParquetError> {
    if fields.is_empty() {
        return Err(ParquetError::EmptySchema);
    }

    let mut arrow_fields = Vec::with_capacity(fields.len());
    for field in fields {
        arrow_fields.push(build_arrow_field(field)?);
    }

    Ok(Schema::new(arrow_fields))
}

fn build_arrow_field(field: &ParquetField) -> Result<Field, ParquetError> {
    let mut data_type = match field.r#type {
        ParquetFieldType::Boolean => DataType::Boolean,
        ParquetFieldType::Int8 => DataType::Int8,
        ParquetFieldType::Int16 => DataType::Int16,
        ParquetFieldType::Int32 => DataType::Int32,
        ParquetFieldType::Int64 => DataType::Int64,
        ParquetFieldType::Float => DataType::Float32,
        ParquetFieldType::Double => DataType::Float64,
        ParquetFieldType::ByteArray => DataType::Binary,
        ParquetFieldType::Utf8 => DataType::Utf8,
        ParquetFieldType::Decimal32 | ParquetFieldType::Decimal64 => {
            if field.decimal_precision == 0 {
                return Err(ParquetError::DecimalPrecisionRequired {
                    field_name: field.name.clone(),
                });
            }

            if field.decimal_scale.abs() as u8 > field.decimal_precision {
                return Err(ParquetError::InvalidDecimal {
                    field_name: field.name.clone(),
                    precision: field.decimal_precision,
                    scale: field.decimal_scale,
                });
            }

            DataType::Decimal128(field.decimal_precision, field.decimal_scale)
        }
        ParquetFieldType::List => {
            let element = field
                .fields
                .first()
                .ok_or_else(|| ParquetError::MissingListChild {
                    field_name: field.name.clone(),
                })?;
            let element_field = build_arrow_field(element)?;
            DataType::List(Arc::new(element_field))
        }
        ParquetFieldType::Map => {
            return Err(ParquetError::UnsupportedMap {
                field_name: field.name.clone(),
            });
        }
    };

    // Parquet repeated leaf fields are represented as lists in Arrow.
    if field.repeated && !matches!(field.r#type, ParquetFieldType::List) {
        let element_field = Field::new("item", data_type, field.optional);
        data_type = DataType::List(Arc::new(element_field));
    }

    let nullable = field.optional || field.repeated;
    Ok(Field::new(&field.name, data_type, nullable))
}

fn build_record_batch(schema: Arc<Schema>, events: &[Event]) -> Result<RecordBatch, ParquetError> {
    let mut columns = Vec::with_capacity(schema.fields().len());

    for field in schema.fields() {
        let column = match field.data_type() {
            DataType::Boolean => build_boolean_array(events, field.name(), field.is_nullable())?,
            DataType::Int8 => build_int8_array(events, field.name(), field.is_nullable())?,
            DataType::Int16 => build_int16_array(events, field.name(), field.is_nullable())?,
            DataType::Int32 => build_int32_array(events, field.name(), field.is_nullable())?,
            DataType::Int64 => build_int64_array(events, field.name(), field.is_nullable())?,
            DataType::Float32 => build_float32_array(events, field.name(), field.is_nullable())?,
            DataType::Float64 => build_float64_array(events, field.name(), field.is_nullable())?,
            DataType::Binary => build_binary_array(events, field.name(), field.is_nullable())?,
            DataType::Utf8 => build_string_array(events, field.name(), field.is_nullable())?,
            DataType::Decimal128(precision, scale) => build_decimal128_array(
                events,
                field.name(),
                *precision,
                *scale,
                field.is_nullable(),
            )?,
            DataType::List(child) => match child.data_type() {
                DataType::Boolean => build_boolean_list_array(
                    events,
                    field.name(),
                    field.is_nullable(),
                    child.is_nullable(),
                )?,
                DataType::Int8 => build_int8_list_array(
                    events,
                    field.name(),
                    field.is_nullable(),
                    child.is_nullable(),
                )?,
                DataType::Int16 => build_int16_list_array(
                    events,
                    field.name(),
                    field.is_nullable(),
                    child.is_nullable(),
                )?,
                DataType::Int32 => build_int32_list_array(
                    events,
                    field.name(),
                    field.is_nullable(),
                    child.is_nullable(),
                )?,
                DataType::Int64 => build_int64_list_array(
                    events,
                    field.name(),
                    field.is_nullable(),
                    child.is_nullable(),
                )?,
                DataType::Float32 => build_float32_list_array(
                    events,
                    field.name(),
                    field.is_nullable(),
                    child.is_nullable(),
                )?,
                DataType::Float64 => build_float64_list_array(
                    events,
                    field.name(),
                    field.is_nullable(),
                    child.is_nullable(),
                )?,
                DataType::Binary => build_binary_list_array(
                    events,
                    field.name(),
                    field.is_nullable(),
                    child.is_nullable(),
                )?,
                DataType::Utf8 => build_string_list_array(
                    events,
                    field.name(),
                    field.is_nullable(),
                    child.is_nullable(),
                )?,
                DataType::Decimal128(precision, scale) => build_decimal128_list_array(
                    events,
                    field.name(),
                    *precision,
                    *scale,
                    field.is_nullable(),
                    child.is_nullable(),
                )?,
                other => {
                    return Err(ParquetError::UnsupportedType {
                        field_name: field.name().to_string(),
                        data_type: other.clone(),
                    });
                }
            },
            other => {
                return Err(ParquetError::UnsupportedType {
                    field_name: field.name().to_string(),
                    data_type: other.clone(),
                });
            }
        };

        columns.push(column);
    }

    RecordBatch::try_new(schema, columns).map_err(|source| ParquetError::ArrowError { source })
}

macro_rules! handle_null_constraints {
    ($builder:expr, $nullable:expr, $field_name:expr) => {{
        if !$nullable {
            return Err(ParquetError::NullConstraint {
                field_name: $field_name.into(),
            });
        }
        $builder.append_null();
    }};
}

macro_rules! define_build_primitive_array_fn {
    (
        $fn_name:ident,
        $builder_ty:ty,
        $( $value_pat:pat $(if $guard:expr)? => $append_expr:expr ),+
    ) => {
        fn $fn_name(
            events: &[Event],
            field_name: &str,
            nullable: bool,
        ) -> Result<ArrayRef, ParquetError> {
            let mut builder = <$builder_ty>::with_capacity(events.len());

            for event in events {
                if let Event::Log(log) = event {
                    match log.get(field_name) {
                        $(
                            Some($value_pat) $(if $guard)? => builder.append_value($append_expr),
                        )+
                        _ => handle_null_constraints!(builder, nullable, field_name),
                    }
                } else {
                    handle_null_constraints!(builder, nullable, field_name);
                }
            }

            Ok(Arc::new(builder.finish()))
        }
    };
}

macro_rules! define_build_list_array_fn {
    (
        $fn_name:ident,
        $builder_ty:ty,
        $( $value_pat:pat $(if $guard:expr)? => $append_expr:expr ),+
    ) => {
        fn $fn_name(
            events: &[Event],
            field_name: &str,
            nullable: bool,
            element_nullable: bool,
        ) -> Result<ArrayRef, ParquetError> {
            let mut builder = ListBuilder::new(<$builder_ty>::with_capacity(events.len()));

            for event in events {
                if let Event::Log(log) = event {
                    match log.get(field_name) {
                        Some(Value::Array(items)) => {
                            for item in items {
                                match item {
                                    $( $value_pat $(if $guard)? => builder.values().append_value($append_expr), )+
                                    _ => {
                                        if !element_nullable {
                                            return Err(ParquetError::NullConstraint {
                                                field_name: field_name.into(),
                                            });
                                        }
                                        builder.values().append_null();
                                    }
                                }
                            }

                            builder.append(true);
                        }
                        Some(value) => {
                            // Allow scalar values for repeated fields as single-element lists.
                            match value {
                                $( $value_pat $(if $guard)? => {
                                    builder.values().append_value($append_expr);
                                    builder.append(true);
                                }, )+
                                _ => {
                                    if !nullable {
                                        return Err(ParquetError::NullConstraint {
                                            field_name: field_name.into(),
                                        });
                                    }
                                    builder.append(false);
                                }
                            }
                        }
                        _ => {
                            if !nullable {
                                return Err(ParquetError::NullConstraint {
                                    field_name: field_name.into(),
                                });
                            }
                            builder.append(false);
                        }
                    }
                } else {
                    if !nullable {
                        return Err(ParquetError::NullConstraint {
                            field_name: field_name.into(),
                        });
                    }
                    builder.append(false);
                }
            }

            Ok(Arc::new(builder.finish()))
        }
    };
}

define_build_primitive_array_fn!(
    build_boolean_array,
    BooleanBuilder,
    Value::Boolean(b) => *b
);

define_build_primitive_array_fn!(
    build_int8_array,
    Int8Builder,
    Value::Integer(i) if *i >= i8::MIN as i64 && *i <= i8::MAX as i64 => *i as i8
);

define_build_primitive_array_fn!(
    build_int16_array,
    Int16Builder,
    Value::Integer(i) if *i >= i16::MIN as i64 && *i <= i16::MAX as i64 => *i as i16
);

define_build_primitive_array_fn!(
    build_int32_array,
    Int32Builder,
    Value::Integer(i) if *i >= i32::MIN as i64 && *i <= i32::MAX as i64 => *i as i32
);

define_build_primitive_array_fn!(
    build_int64_array,
    Int64Builder,
    Value::Integer(i) => *i
);

define_build_primitive_array_fn!(
    build_float32_array,
    Float32Builder,
    Value::Float(f) => f.into_inner() as f32,
    Value::Integer(i) => *i as f32
);

define_build_primitive_array_fn!(
    build_float64_array,
    Float64Builder,
    Value::Float(f) => f.into_inner(),
    Value::Integer(i) => *i as f64
);

define_build_list_array_fn!(
    build_boolean_list_array,
    BooleanBuilder,
    Value::Boolean(b) => *b
);

define_build_list_array_fn!(
    build_int8_list_array,
    Int8Builder,
    Value::Integer(i) if *i >= i8::MIN as i64 && *i <= i8::MAX as i64 => *i as i8
);

define_build_list_array_fn!(
    build_int16_list_array,
    Int16Builder,
    Value::Integer(i) if *i >= i16::MIN as i64 && *i <= i16::MAX as i64 => *i as i16
);

define_build_list_array_fn!(
    build_int32_list_array,
    Int32Builder,
    Value::Integer(i) if *i >= i32::MIN as i64 && *i <= i32::MAX as i64 => *i as i32
);

define_build_list_array_fn!(
    build_int64_list_array,
    Int64Builder,
    Value::Integer(i) => *i
);

define_build_list_array_fn!(
    build_float32_list_array,
    Float32Builder,
    Value::Float(f) => f.into_inner() as f32,
    Value::Integer(i) => *i as f32
);

define_build_list_array_fn!(
    build_float64_list_array,
    Float64Builder,
    Value::Float(f) => f.into_inner(),
    Value::Integer(i) => *i as f64
);

fn build_string_array(
    events: &[Event],
    field_name: &str,
    nullable: bool,
) -> Result<ArrayRef, ParquetError> {
    let mut builder = StringBuilder::with_capacity(events.len(), 0);

    for event in events {
        if let Event::Log(log) = event {
            match log.get(field_name) {
                Some(Value::Bytes(bytes)) => {
                    match std::str::from_utf8(bytes) {
                        Ok(s) => builder.append_value(s),
                        Err(_) => builder.append_value(&String::from_utf8_lossy(bytes)),
                    };
                }
                Some(value) => builder.append_value(&value.to_string_lossy()),
                _ => handle_null_constraints!(builder, nullable, field_name),
            }
        } else {
            handle_null_constraints!(builder, nullable, field_name);
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn build_binary_array(
    events: &[Event],
    field_name: &str,
    nullable: bool,
) -> Result<ArrayRef, ParquetError> {
    let mut builder = BinaryBuilder::with_capacity(events.len(), 0);

    for event in events {
        if let Event::Log(log) = event {
            match log.get(field_name) {
                Some(Value::Bytes(bytes)) => builder.append_value(bytes),
                _ => handle_null_constraints!(builder, nullable, field_name),
            }
        } else {
            handle_null_constraints!(builder, nullable, field_name);
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn build_decimal128_array(
    events: &[Event],
    field_name: &str,
    precision: u8,
    scale: i8,
    nullable: bool,
) -> Result<ArrayRef, ParquetError> {
    let mut builder = Decimal128Builder::with_capacity(events.len())
        .with_precision_and_scale(precision, scale)
        .map_err(|source| ParquetError::ArrowError { source })?;

    let target_scale = scale.unsigned_abs() as u32;

    for event in events {
        if let Event::Log(log) = event {
            match log.get(field_name) {
                Some(Value::Float(f)) => {
                    if let Ok(mut decimal) = Decimal::try_from(f.into_inner()) {
                        decimal.rescale(target_scale);
                        builder.append_value(decimal.mantissa());
                    } else if !nullable {
                        return Err(ParquetError::NullConstraint {
                            field_name: field_name.into(),
                        });
                    } else {
                        builder.append_null();
                    }
                }
                Some(Value::Integer(i)) => {
                    let mut decimal = Decimal::from(*i);
                    decimal.rescale(target_scale);
                    builder.append_value(decimal.mantissa());
                }
                Some(Value::Bytes(bytes)) => {
                    if let Ok(s) = std::str::from_utf8(bytes) {
                        if let Ok(mut decimal) = Decimal::from_str_exact(s) {
                            decimal.rescale(target_scale);
                            builder.append_value(decimal.mantissa());
                        } else if !nullable {
                            return Err(ParquetError::NullConstraint {
                                field_name: field_name.into(),
                            });
                        } else {
                            builder.append_null();
                        }
                    } else if !nullable {
                        return Err(ParquetError::NullConstraint {
                            field_name: field_name.into(),
                        });
                    } else {
                        builder.append_null();
                    }
                }
                _ => handle_null_constraints!(builder, nullable, field_name),
            }
        } else {
            handle_null_constraints!(builder, nullable, field_name);
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn build_string_list_array(
    events: &[Event],
    field_name: &str,
    nullable: bool,
    element_nullable: bool,
) -> Result<ArrayRef, ParquetError> {
    let mut builder = ListBuilder::new(StringBuilder::with_capacity(events.len(), 0));

    for event in events {
        if let Event::Log(log) = event {
            match log.get(field_name) {
                Some(Value::Array(items)) => {
                    for item in items {
                        match item {
                            Value::Bytes(bytes) => match std::str::from_utf8(bytes) {
                                Ok(s) => builder.values().append_value(s),
                                Err(_) => builder
                                    .values()
                                    .append_value(&String::from_utf8_lossy(bytes)),
                            },
                            other => {
                                if !element_nullable {
                                    return Err(ParquetError::NullConstraint {
                                        field_name: field_name.into(),
                                    });
                                }
                                builder.values().append_value(&other.to_string_lossy());
                            }
                        }
                    }

                    builder.append(true);
                }
                Some(Value::Bytes(bytes)) => {
                    match std::str::from_utf8(bytes) {
                        Ok(s) => builder.values().append_value(s),
                        Err(_) => builder
                            .values()
                            .append_value(&String::from_utf8_lossy(bytes)),
                    }
                    builder.append(true);
                }
                Some(_other) => {
                    if !nullable {
                        return Err(ParquetError::NullConstraint {
                            field_name: field_name.into(),
                        });
                    }
                    if element_nullable {
                        builder.values().append_null();
                    }
                    builder.append(false);
                }
                _ => {
                    if !nullable {
                        return Err(ParquetError::NullConstraint {
                            field_name: field_name.into(),
                        });
                    }
                    builder.append(false);
                }
            }
        } else if !nullable {
            return Err(ParquetError::NullConstraint {
                field_name: field_name.into(),
            });
        } else {
            builder.append(false);
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn build_binary_list_array(
    events: &[Event],
    field_name: &str,
    nullable: bool,
    element_nullable: bool,
) -> Result<ArrayRef, ParquetError> {
    let mut builder = ListBuilder::new(BinaryBuilder::with_capacity(events.len(), 0));

    for event in events {
        if let Event::Log(log) = event {
            match log.get(field_name) {
                Some(Value::Array(items)) => {
                    for item in items {
                        match item {
                            Value::Bytes(bytes) => builder.values().append_value(bytes),
                            _ if element_nullable => builder.values().append_null(),
                            _ => {
                                return Err(ParquetError::NullConstraint {
                                    field_name: field_name.into(),
                                });
                            }
                        }
                    }

                    builder.append(true);
                }
                Some(Value::Bytes(bytes)) => {
                    builder.values().append_value(bytes);
                    builder.append(true);
                }
                _ => {
                    if !nullable {
                        return Err(ParquetError::NullConstraint {
                            field_name: field_name.into(),
                        });
                    }
                    builder.append(false);
                }
            }
        } else if !nullable {
            return Err(ParquetError::NullConstraint {
                field_name: field_name.into(),
            });
        } else {
            builder.append(false);
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn build_decimal128_list_array(
    events: &[Event],
    field_name: &str,
    precision: u8,
    scale: i8,
    nullable: bool,
    element_nullable: bool,
) -> Result<ArrayRef, ParquetError> {
    let value_builder = Decimal128Builder::with_capacity(events.len())
        .with_precision_and_scale(precision, scale)
        .map_err(|source| ParquetError::ArrowError { source })?;
    let mut builder = ListBuilder::new(value_builder);
    let target_scale = scale.unsigned_abs() as u32;

    for event in events {
        if let Event::Log(log) = event {
            match log.get(field_name) {
                Some(Value::Array(items)) => {
                    for item in items {
                        match item {
                            Value::Float(f) => {
                                if let Ok(mut decimal) = Decimal::try_from(f.into_inner()) {
                                    decimal.rescale(target_scale);
                                    builder.values().append_value(decimal.mantissa());
                                } else if !element_nullable {
                                    return Err(ParquetError::NullConstraint {
                                        field_name: field_name.into(),
                                    });
                                } else {
                                    builder.values().append_null();
                                }
                            }
                            Value::Integer(i) => {
                                let mut decimal = Decimal::from(*i);
                                decimal.rescale(target_scale);
                                builder.values().append_value(decimal.mantissa());
                            }
                            Value::Bytes(bytes) => {
                                if let Ok(s) = std::str::from_utf8(bytes) {
                                    if let Ok(mut decimal) = Decimal::from_str_exact(s) {
                                        decimal.rescale(target_scale);
                                        builder.values().append_value(decimal.mantissa());
                                    } else if !element_nullable {
                                        return Err(ParquetError::NullConstraint {
                                            field_name: field_name.into(),
                                        });
                                    } else {
                                        builder.values().append_null();
                                    }
                                } else if !element_nullable {
                                    return Err(ParquetError::NullConstraint {
                                        field_name: field_name.into(),
                                    });
                                } else {
                                    builder.values().append_null();
                                }
                            }
                            _ if element_nullable => builder.values().append_null(),
                            _ => {
                                return Err(ParquetError::NullConstraint {
                                    field_name: field_name.into(),
                                });
                            }
                        }
                    }

                    builder.append(true);
                }
                Some(value) => {
                    match value {
                        Value::Float(f) => {
                            if let Ok(mut decimal) = Decimal::try_from(f.into_inner()) {
                                decimal.rescale(target_scale);
                                builder.values().append_value(decimal.mantissa());
                            } else if !nullable {
                                return Err(ParquetError::NullConstraint {
                                    field_name: field_name.into(),
                                });
                            } else {
                                builder.values().append_null();
                            }
                        }
                        Value::Integer(i) => {
                            let mut decimal = Decimal::from(*i);
                            decimal.rescale(target_scale);
                            builder.values().append_value(decimal.mantissa());
                        }
                        Value::Bytes(bytes) => {
                            if let Ok(s) = std::str::from_utf8(bytes) {
                                if let Ok(mut decimal) = Decimal::from_str_exact(s) {
                                    decimal.rescale(target_scale);
                                    builder.values().append_value(decimal.mantissa());
                                } else if !nullable {
                                    return Err(ParquetError::NullConstraint {
                                        field_name: field_name.into(),
                                    });
                                } else {
                                    builder.values().append_null();
                                }
                            } else if !nullable {
                                return Err(ParquetError::NullConstraint {
                                    field_name: field_name.into(),
                                });
                            } else {
                                builder.values().append_null();
                            }
                        }
                        _ if element_nullable => builder.values().append_null(),
                        _ if !nullable => {
                            return Err(ParquetError::NullConstraint {
                                field_name: field_name.into(),
                            });
                        }
                        _ => builder.values().append_null(),
                    }

                    builder.append(true);
                }
                _ => {
                    if !nullable {
                        return Err(ParquetError::NullConstraint {
                            field_name: field_name.into(),
                        });
                    }
                    builder.append(false);
                }
            }
        } else if !nullable {
            return Err(ParquetError::NullConstraint {
                field_name: field_name.into(),
            });
        } else {
            builder.append(false);
        }
    }

    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Array, BinaryArray, BooleanArray, Decimal128Array, Float64Array, Int64Array, ListArray,
        StringArray,
    };
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::{fs::File, path::PathBuf};
    use vector_core::event::LogEvent;
    use vrl::btreemap;

    fn test_schema() -> Vec<ParquetField> {
        vec![
            ParquetField {
                name: "foo".to_string(),
                r#type: ParquetFieldType::Int64,
                decimal_precision: 0,
                decimal_scale: 0,
                repeated: false,
                optional: false,
                fields: Vec::new(),
            },
            ParquetField {
                name: "bar".to_string(),
                r#type: ParquetFieldType::Utf8,
                decimal_precision: 0,
                decimal_scale: 0,
                repeated: false,
                optional: true,
                fields: Vec::new(),
            },
        ]
    }

    fn build_serializer(schema: Vec<ParquetField>) -> ParquetSerializer {
        let config = ParquetSerializerConfig {
            parquet: ParquetSerializerOptions {
                schema,
                default_compression: ParquetCompression::Uncompressed,
                default_encoding: ParquetEncoding::Plain,
            },
        };

        config.build().expect("build serializer")
    }

    fn read_parquet_batch(bytes: &BytesMut) -> RecordBatch {
        let bytes = bytes.clone().freeze();
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .expect("create Parquet reader builder");
        let mut reader = builder.build().expect("build Parquet reader");
        let batch = reader.next().expect("read batch").expect("decode batch");
        assert!(reader.next().is_none());
        batch
    }

    #[test]
    fn serialize_parquet() {
        let events = vec![
            Event::Log(LogEvent::from(btreemap! {
                "foo" => Value::from(42),
                "bar" => Value::from("baz")
            })),
            Event::Log(LogEvent::from(btreemap! {
                "foo" => Value::from(7),
                "bar" => Value::from("quux")
            })),
        ];

        let mut serializer = build_serializer(test_schema());
        let mut bytes = BytesMut::new();

        serializer.encode(events[0].clone(), &mut bytes).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn parquet_primitives_round_trip() {
        let schema = vec![
            ParquetField {
                name: "flag".to_string(),
                r#type: ParquetFieldType::Boolean,
                decimal_precision: 0,
                decimal_scale: 0,
                repeated: false,
                optional: false,
                fields: Vec::new(),
            },
            ParquetField {
                name: "count".to_string(),
                r#type: ParquetFieldType::Int64,
                decimal_precision: 0,
                decimal_scale: 0,
                repeated: false,
                optional: false,
                fields: Vec::new(),
            },
            ParquetField {
                name: "ratio".to_string(),
                r#type: ParquetFieldType::Double,
                decimal_precision: 0,
                decimal_scale: 0,
                repeated: false,
                optional: false,
                fields: Vec::new(),
            },
            ParquetField {
                name: "message".to_string(),
                r#type: ParquetFieldType::Utf8,
                decimal_precision: 0,
                decimal_scale: 0,
                repeated: false,
                optional: true,
                fields: Vec::new(),
            },
            ParquetField {
                name: "payload".to_string(),
                r#type: ParquetFieldType::ByteArray,
                decimal_precision: 0,
                decimal_scale: 0,
                repeated: false,
                optional: true,
                fields: Vec::new(),
            },
            ParquetField {
                name: "money".to_string(),
                r#type: ParquetFieldType::Decimal64,
                decimal_precision: 6,
                decimal_scale: 2,
                repeated: false,
                optional: false,
                fields: Vec::new(),
            },
        ];

        let events = vec![
            Event::Log(LogEvent::from(btreemap! {
                "flag" => Value::from(true),
                "count" => Value::from(123i64),
                "ratio" => Value::from(3.14),
                "message" => Value::from("hello"),
                "payload" => Value::Bytes(vec![1u8, 2, 3, 4].into()),
                "money" => Value::Bytes("12.34".into()),
            })),
            Event::Log(LogEvent::from(btreemap! {
                "flag" => Value::from(false),
                "count" => Value::from(0i64),
                "ratio" => Value::from(0.5),
                "payload" => Value::Bytes(Vec::new().into()),
                "money" => Value::from(5678i64),
            })),
        ];

        let serializer = build_serializer(schema);
        let mut bytes = BytesMut::new();

        serializer
            .encode_events(&events, &mut bytes)
            .expect("encode events");
        let batch = read_parquet_batch(&bytes);

        let flags = batch
            .column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("bool column");
        assert!(flags.value(0));
        assert!(!flags.value(1));

        let counts = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count column");
        assert_eq!(counts.value(0), 123);
        assert_eq!(counts.value(1), 0);

        let ratios = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("ratio column");
        assert!((ratios.value(0) - 3.14).abs() < f64::EPSILON);
        assert!((ratios.value(1) - 0.5).abs() < f64::EPSILON);

        let messages = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("message column");
        assert_eq!(messages.value(0), "hello");
        assert!(messages.is_null(1));

        let payloads = batch
            .column(4)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .expect("payload column");
        assert_eq!(payloads.value(0), [1, 2, 3, 4]);
        assert!(payloads.value(1).is_empty());

        let decimals = batch
            .column(5)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal column");
        assert_eq!(
            Decimal::from_i128_with_scale(decimals.value(0), 2).to_string(),
            "12.34"
        );
        assert_eq!(
            Decimal::from_i128_with_scale(decimals.value(1), 2).to_string(),
            "5678.00"
        );
    }

    #[test]
    fn parquet_repeated_fields_round_trip() {
        let schema = vec![
            ParquetField {
                name: "ids".to_string(),
                r#type: ParquetFieldType::Int64,
                decimal_precision: 0,
                decimal_scale: 0,
                repeated: true,
                optional: true,
                fields: Vec::new(),
            },
            ParquetField {
                name: "precise".to_string(),
                r#type: ParquetFieldType::Decimal64,
                decimal_precision: 6,
                decimal_scale: 3,
                repeated: true,
                optional: true,
                fields: Vec::new(),
            },
            ParquetField {
                name: "names".to_string(),
                r#type: ParquetFieldType::Utf8,
                decimal_precision: 0,
                decimal_scale: 0,
                repeated: true,
                optional: true,
                fields: Vec::new(),
            },
        ];

        let events = vec![
            Event::Log(LogEvent::from(btreemap! {
                "ids" => Value::Array(vec![1.into(), 2.into(), 3.into()]),
                "precise" => Value::Array(vec![
                    Value::Bytes("1.001".into()),
                    Value::Integer(2),
                ]),
                "names" => Value::Array(vec![Value::from("alice"), Value::from("bob")]),
            })),
            Event::Log(LogEvent::from(btreemap! {
                "ids" => Value::from(99),
                "precise" => Value::Bytes("3.500".into()),
            })),
        ];

        let serializer = build_serializer(schema);
        let mut bytes = BytesMut::new();

        serializer
            .encode_events(&events, &mut bytes)
            .expect("encode events");
        let batch = read_parquet_batch(&bytes);

        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("ids column");
        let id_values = ids.values().as_any().downcast_ref::<Int64Array>().unwrap();
        let first_ids_ref = ids.value(0);
        let first_ids = first_ids_ref
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("first id list");
        assert_eq!(first_ids.values(), &[1, 2, 3]);
        let second_ids_ref = ids.value(1);
        let second_ids = second_ids_ref
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("second id list");
        assert_eq!(second_ids.values(), &[99]);
        assert_eq!(id_values.len(), 4);

        let decimals = batch
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("precise column");
        let first_decimal_values_ref = decimals.value(0);
        let first_decimal_values = first_decimal_values_ref
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("first decimal list");
        let first_decimal_strings: Vec<String> = (0..first_decimal_values.len())
            .map(|idx| {
                Decimal::from_i128_with_scale(first_decimal_values.value(idx), 3).to_string()
            })
            .collect();
        assert_eq!(first_decimal_strings, ["1.001", "2.000"]);

        let second_decimal_values_ref = decimals.value(1);
        let second_decimal_values = second_decimal_values_ref
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("second decimal list");
        assert_eq!(
            Decimal::from_i128_with_scale(second_decimal_values.value(0), 3).to_string(),
            "3.500"
        );

        let names = batch
            .column(2)
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("names column");
        let first_names_ref = names.value(0);
        let first_names = first_names_ref
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("first name list");
        assert_eq!(first_names.value(0), "alice");
        assert_eq!(first_names.value(1), "bob");
        assert!(names.is_null(1));
    }

    #[test]
    fn parquet_fixture_zstd_round_trip() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/data/test-zst.parquet");
        let file = File::open(&path).expect("open parquet fixture");
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).expect("create parquet reader builder");
        let mut reader = builder.build().expect("build parquet reader");

        let batch = reader.next().expect("read batch").expect("decode batch");
        assert!(reader.next().is_none());

        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("ids column");
        let trees = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("tree column");

        assert_eq!(ids.len(), 7);
        assert_eq!(trees.len(), 7);

        let expected = ["oak", "ash", "yew", "beech", "yew", "beech", "yew"];
        for idx in 0..ids.len() {
            assert_eq!(ids.value(idx), (idx + 1) as i64);
            assert_eq!(trees.value(idx), expected[idx]);
        }
    }
}

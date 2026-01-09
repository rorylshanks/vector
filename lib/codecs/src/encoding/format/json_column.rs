//! JSON column expansion for Parquet encoding
//!
//! This module provides functionality to expand JSON string columns into
//! individual subcolumns for efficient columnar storage, similar to ClickHouse's JSON type.
//!
//! Performance optimizations:
//! - Two-pass approach to minimize memory usage
//! - AHashMap for faster internal hashing
//! - MurmurHash3 for bucket assignment (consistent hashing)
//! - Parallel processing with rayon
//! - String interning to avoid allocations

use std::collections::BTreeMap;
use std::sync::Arc;

use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;
use smallvec::SmallVec;

use arrow::array::{
    ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, LargeStringBuilder, MapBuilder,
    UInt64Builder,
};

use super::parquet::{JsonColumnConfig, JsonTypeHint};

/// Compact representation of a JSON value for efficient storage
#[derive(Debug, Clone)]
enum CompactValue {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    String(String),
}

impl CompactValue {
    /// Convert from simd_json borrowed value (zero-copy for primitives, copy only for strings)
    #[inline]
    fn from_simd_borrowed(value: &simd_json::BorrowedValue<'_>) -> Self {
        use simd_json::BorrowedValue;
        match value {
            BorrowedValue::Static(simd_json::StaticNode::Null) => CompactValue::Null,
            BorrowedValue::Static(simd_json::StaticNode::Bool(b)) => CompactValue::Bool(*b),
            BorrowedValue::Static(simd_json::StaticNode::I64(i)) => CompactValue::Int(*i),
            BorrowedValue::Static(simd_json::StaticNode::U64(u)) => {
                if *u > i64::MAX as u64 {
                    CompactValue::String(u.to_string())
                } else {
                    CompactValue::Uint(*u)
                }
            }
            BorrowedValue::Static(simd_json::StaticNode::F64(f)) => CompactValue::Float(*f),
            BorrowedValue::String(s) => CompactValue::String(s.to_string()),
            BorrowedValue::Array(_) | BorrowedValue::Object(_) => {
                CompactValue::String(simd_json::to_string(value).unwrap_or_default())
            }
        }
    }

    #[inline]
    fn is_null_or_empty(&self) -> bool {
        match self {
            CompactValue::Null => true,
            CompactValue::String(s) => s.is_empty(),
            _ => false,
        }
    }

    #[inline]
    fn to_string_value(&self) -> String {
        match self {
            CompactValue::Null => String::new(),
            CompactValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            CompactValue::Int(i) => i.to_string(),
            CompactValue::Uint(u) => u.to_string(),
            CompactValue::Float(f) => f.to_string(),
            CompactValue::String(s) => s.clone(),
        }
    }

    #[inline]
    fn as_i64(&self) -> Option<i64> {
        match self {
            CompactValue::Int(i) => Some(*i),
            CompactValue::Uint(u) if *u <= i64::MAX as u64 => Some(*u as i64),
            CompactValue::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    #[inline]
    fn as_u64(&self) -> Option<u64> {
        match self {
            CompactValue::Uint(u) => Some(*u),
            CompactValue::Int(i) if *i >= 0 => Some(*i as u64),
            CompactValue::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    #[inline]
    fn as_f64(&self) -> Option<f64> {
        match self {
            CompactValue::Float(f) => Some(*f),
            CompactValue::Int(i) => Some(*i as f64),
            CompactValue::Uint(u) => Some(*u as f64),
            CompactValue::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    #[inline]
    fn as_bool(&self) -> Option<bool> {
        match self {
            CompactValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Result of processing JSON columns for a batch of events
#[derive(Debug)]
pub struct ProcessedJsonColumns {
    /// The original column name
    pub column_name: String,
    /// Original JSON string values (if keep_original_column is true)
    pub original_values: Option<Vec<Option<String>>>,
    /// Subcolumns with their flattened keys and values
    pub subcolumns: BTreeMap<String, SubcolumnData>,
    /// Bucket maps for overflow keys - sparse storage: bucket_index -> (row_index -> (key -> value))
    pub bucket_maps: Vec<AHashMap<usize, AHashMap<String, String>>>,
    /// Number of buckets
    pub bucket_count: usize,
    /// Total number of rows (needed for generating arrays)
    pub num_rows: usize,
}

/// Data for a single subcolumn
#[derive(Debug)]
pub enum SubcolumnData {
    String(Vec<Option<String>>),
    Int64(Vec<Option<i64>>),
    Uint64(Vec<Option<u64>>),
    Float64(Vec<Option<f64>>),
    Boolean(Vec<Option<bool>>),
}

impl SubcolumnData {
    /// Convert this subcolumn data to an Arrow array
    pub fn to_arrow_array(&self) -> ArrayRef {
        match self {
            SubcolumnData::String(values) => {
                let mut builder = LargeStringBuilder::with_capacity(values.len(), values.len() * 32);
                for value in values {
                    match value {
                        Some(v) => builder.append_value(v),
                        None => builder.append_null(),
                    }
                }
                Arc::new(builder.finish())
            }
            SubcolumnData::Int64(values) => {
                let mut builder = Int64Builder::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(v) => builder.append_value(*v),
                        None => builder.append_null(),
                    }
                }
                Arc::new(builder.finish())
            }
            SubcolumnData::Uint64(values) => {
                let mut builder = UInt64Builder::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(v) => builder.append_value(*v),
                        None => builder.append_null(),
                    }
                }
                Arc::new(builder.finish())
            }
            SubcolumnData::Float64(values) => {
                let mut builder = Float64Builder::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(v) => builder.append_value(*v),
                        None => builder.append_null(),
                    }
                }
                Arc::new(builder.finish())
            }
            SubcolumnData::Boolean(values) => {
                let mut builder = BooleanBuilder::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(v) => builder.append_value(*v),
                        None => builder.append_null(),
                    }
                }
                Arc::new(builder.finish())
            }
        }
    }
}

/// Inferred type for a JSON value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferredType {
    Null,
    Boolean,
    Int64,
    Uint64,
    Float64,
    String,
}

impl InferredType {
    /// Resolve type conflicts by finding a common type
    #[inline]
    fn resolve_conflict(self, other: InferredType) -> InferredType {
        use InferredType::*;
        match (self, other) {
            (Null, other) | (other, Null) => other,
            (a, b) if a == b => a,
            (Int64, Uint64) | (Uint64, Int64) => Int64,
            (Float64, Int64 | Uint64) | (Int64 | Uint64, Float64) => Float64,
            _ => String,
        }
    }
}

/// Thread-safe builder for subcolumn data
enum SubcolumnBuilder {
    String(Vec<Option<String>>),
    Int64(Vec<Option<i64>>),
    Uint64(Vec<Option<u64>>),
    Float64(Vec<Option<f64>>),
    Boolean(Vec<Option<bool>>),
}

impl SubcolumnBuilder {
    fn new(inferred_type: InferredType, capacity: usize) -> Self {
        match inferred_type {
            InferredType::String | InferredType::Null => {
                SubcolumnBuilder::String(Vec::with_capacity(capacity))
            }
            InferredType::Int64 => SubcolumnBuilder::Int64(Vec::with_capacity(capacity)),
            InferredType::Uint64 => SubcolumnBuilder::Uint64(Vec::with_capacity(capacity)),
            InferredType::Float64 => SubcolumnBuilder::Float64(Vec::with_capacity(capacity)),
            InferredType::Boolean => SubcolumnBuilder::Boolean(Vec::with_capacity(capacity)),
        }
    }

    #[inline]
    fn push(&mut self, value: Option<&CompactValue>) {
        match self {
            SubcolumnBuilder::String(vec) => {
                vec.push(value.map(|v| v.to_string_value()));
            }
            SubcolumnBuilder::Int64(vec) => {
                vec.push(value.and_then(|v| v.as_i64()));
            }
            SubcolumnBuilder::Uint64(vec) => {
                vec.push(value.and_then(|v| v.as_u64()));
            }
            SubcolumnBuilder::Float64(vec) => {
                vec.push(value.and_then(|v| v.as_f64()));
            }
            SubcolumnBuilder::Boolean(vec) => {
                vec.push(value.and_then(|v| v.as_bool()));
            }
        }
    }

    fn finish(self) -> SubcolumnData {
        match self {
            SubcolumnBuilder::String(vec) => SubcolumnData::String(vec),
            SubcolumnBuilder::Int64(vec) => SubcolumnData::Int64(vec),
            SubcolumnBuilder::Uint64(vec) => SubcolumnData::Uint64(vec),
            SubcolumnBuilder::Float64(vec) => SubcolumnData::Float64(vec),
            SubcolumnBuilder::Boolean(vec) => SubcolumnData::Boolean(vec),
        }
    }
}

/// Statistics for a single key during processing
struct KeyStats {
    count: usize,
    inferred_type: InferredType,
}

/// Result of parsing a single row in Pass 2
struct RowParseResult {
    /// Values for subcolumns: (builder_index, value)
    subcolumn_values: SmallVec<[(usize, CompactValue); 16]>,
    /// Overflow entries: (bucket_idx, key_idx, value) - using key_idx to avoid String clone
    overflow_entries: SmallVec<[(usize, usize, CompactValue); 8]>,
    /// Original JSON if keeping
    original: Option<String>,
}

/// Processor for expanding JSON columns
#[derive(Clone, Debug)]
pub struct JsonColumnProcessor {
    config: JsonColumnConfig,
}

impl JsonColumnProcessor {
    /// Create a new JSON column processor with the given configuration
    pub fn new(config: JsonColumnConfig) -> Self {
        Self { config }
    }

    /// Get the column name this processor handles
    pub fn column_name(&self) -> &str {
        &self.config.column
    }

    /// Process a batch of JSON string values and return the expanded columns
    ///
    /// Optimized two-pass approach:
    /// - Pass 1: Parallel key scanning to determine schema
    /// - Pass 2: Parallel parsing with direct output building
    pub fn process_batch<'a, I>(&self, json_values: I) -> ProcessedJsonColumns
    where
        I: Iterator<Item = Option<&'a str>>,
    {
        let json_values: Vec<Option<&str>> = json_values.collect();
        let num_rows = json_values.len();
        let max_depth = self.config.max_depth;
        let bucket_count = self.config.bucket_count;
        let max_subcolumns = self.config.max_subcolumns;
        let keep_original = self.config.keep_original_column;

        // ============================================================
        // PASS 1: Parallel key scanning (minimal memory - only keys and types)
        // Uses BorrowedValue to avoid allocating string copies
        // ============================================================
        let key_scan_results: Vec<SmallVec<[(String, InferredType); 32]>> = json_values
            .par_iter()
            .map(|json_str| {
                let mut keys = SmallVec::new();
                if let Some(s) = json_str {
                    if !s.is_empty() {
                        let mut s_owned = s.to_string();
                        if let Ok(simd_json::BorrowedValue::Object(ref obj)) =
                            unsafe { simd_json::from_str::<simd_json::BorrowedValue>(&mut s_owned) }
                        {
                            let mut key_buffer = String::with_capacity(64);
                            Self::scan_keys(obj, "", 0, max_depth, &mut keys, &mut key_buffer);
                        }
                    }
                }
                keys
            })
            .collect();

        // Aggregate key statistics (sequential - fast with AHashMap)
        let mut key_to_idx: AHashMap<String, usize> = AHashMap::with_capacity(256);
        let mut keys: Vec<String> = Vec::with_capacity(256);
        let mut key_stats: Vec<KeyStats> = Vec::with_capacity(256);

        for row_keys in &key_scan_results {
            for (key, inferred_type) in row_keys {
                let key_idx = match key_to_idx.get(key) {
                    Some(&idx) => idx,
                    None => {
                        let idx = keys.len();
                        key_to_idx.insert(key.clone(), idx);
                        keys.push(key.clone());
                        key_stats.push(KeyStats {
                            count: 0,
                            inferred_type: InferredType::Null,
                        });
                        idx
                    }
                };

                if *inferred_type != InferredType::Null {
                    let stats = &mut key_stats[key_idx];
                    stats.count += 1;
                    stats.inferred_type = stats.inferred_type.resolve_conflict(*inferred_type);
                }
            }
        }

        // Free scan results memory
        drop(key_scan_results);

        // Sort keys by frequency, determine subcolumns vs overflow
        let mut sorted_indices: Vec<usize> = (0..keys.len()).collect();
        sorted_indices.sort_unstable_by(|&a, &b| {
            key_stats[b].count.cmp(&key_stats[a].count)
                .then_with(|| keys[a].cmp(&keys[b]))
        });

        let subcolumn_indices: AHashSet<usize> = sorted_indices
            .iter()
            .take(max_subcolumns)
            .copied()
            .collect();

        let overflow_indices: AHashSet<usize> = sorted_indices
            .iter()
            .skip(max_subcolumns)
            .copied()
            .collect();

        // Precompute bucket indices for overflow keys using MurmurHash3
        let mut key_to_bucket: Vec<usize> = vec![0; keys.len()];
        for &idx in &overflow_indices {
            key_to_bucket[idx] = Self::hash_key_to_bucket(&keys[idx], bucket_count);
        }

        // Create mapping from key index to builder index
        let mut key_to_builder: Vec<Option<usize>> = vec![None; keys.len()];
        let subcolumn_keys: Vec<usize> = sorted_indices.iter().take(max_subcolumns).copied().collect();
        for (builder_idx, &key_idx) in subcolumn_keys.iter().enumerate() {
            key_to_builder[key_idx] = Some(builder_idx);
        }

        // ============================================================
        // PASS 2: Chunked parallel parsing with incremental output building
        // Processes rows in chunks to limit peak memory usage
        // ============================================================
        let num_builders = subcolumn_keys.len();

        // Initialize builders
        let mut subcolumn_builders: Vec<SubcolumnBuilder> = subcolumn_keys
            .iter()
            .map(|&key_idx| {
                let final_type = self.get_type_for_key(&keys[key_idx], key_stats[key_idx].inferred_type);
                SubcolumnBuilder::new(final_type, num_rows)
            })
            .collect();

        // Initialize bucket maps
        let mut bucket_maps: Vec<AHashMap<usize, AHashMap<String, String>>> =
            (0..bucket_count).map(|_| AHashMap::new()).collect();

        // Original values
        let mut original_values: Vec<Option<String>> = if keep_original {
            Vec::with_capacity(num_rows)
        } else {
            Vec::new()
        };

        // Temporary storage for row processing
        let mut row_values: Vec<Option<CompactValue>> = vec![None; num_builders];
        let mut row_has_good_value: Vec<bool> = vec![false; num_builders];

        // Process in chunks to limit peak memory (chunk size trades off memory vs parallelism overhead)
        const CHUNK_SIZE: usize = 500;
        let mut global_row_idx = 0usize;

        for chunk in json_values.chunks(CHUNK_SIZE) {
            // Parse chunk in parallel using BorrowedValue
            let chunk_results: Vec<RowParseResult> = chunk
                .par_iter()
                .map(|json_str| {
                    let mut result = RowParseResult {
                        subcolumn_values: SmallVec::new(),
                        overflow_entries: SmallVec::new(),
                        original: None,
                    };

                    match json_str {
                        Some(s) if !s.is_empty() => {
                            if keep_original {
                                result.original = Some((*s).to_string());
                            }

                            let mut s_owned = s.to_string();
                            if let Ok(simd_json::BorrowedValue::Object(ref obj)) =
                                unsafe { simd_json::from_str::<simd_json::BorrowedValue>(&mut s_owned) }
                            {
                                let mut key_buffer = String::with_capacity(64);
                                Self::extract_values(
                                    obj,
                                    "",
                                    0,
                                    max_depth,
                                    &mut key_buffer,
                                    &key_to_idx,
                                    &key_to_builder,
                                    &subcolumn_indices,
                                    &overflow_indices,
                                    &keys,
                                    &key_to_bucket,
                                    &mut result,
                                );
                            }
                        }
                        _ => {}
                    }

                    result
                })
                .collect();

            // Process chunk results sequentially and immediately free memory
            for parse_result in chunk_results {
                // Store original
                if keep_original {
                    original_values.push(parse_result.original);
                }

                // Reset row state
                row_values.iter_mut().for_each(|x| *x = None);
                row_has_good_value.iter_mut().for_each(|x| *x = false);

                // Collect subcolumn values (first non-null wins)
                for (builder_idx, value) in parse_result.subcolumn_values {
                    let is_good = !value.is_null_or_empty();
                    if row_values[builder_idx].is_none() || (!row_has_good_value[builder_idx] && is_good) {
                        row_values[builder_idx] = Some(value);
                        row_has_good_value[builder_idx] = is_good;
                    }
                }

                // Push to builders
                for (builder_idx, builder) in subcolumn_builders.iter_mut().enumerate() {
                    builder.push(row_values[builder_idx].as_ref());
                }

                // Handle overflow entries - look up key by index
                for (bucket_idx, key_idx, value) in parse_result.overflow_entries {
                    bucket_maps[bucket_idx]
                        .entry(global_row_idx)
                        .or_default()
                        .insert(keys[key_idx].clone(), value.to_string_value());
                }

                global_row_idx += 1;
            }
            // chunk_results is dropped here, freeing memory before processing next chunk
        }

        // Finalize subcolumns
        let mut subcolumns: BTreeMap<String, SubcolumnData> = BTreeMap::new();
        for (builder_idx, builder) in subcolumn_builders.into_iter().enumerate() {
            let key_idx = subcolumn_keys[builder_idx];
            let full_key = format!("{}.{}", self.config.column, keys[key_idx]);
            subcolumns.insert(full_key, builder.finish());
        }

        ProcessedJsonColumns {
            column_name: self.config.column.clone(),
            original_values: if keep_original { Some(original_values) } else { None },
            subcolumns,
            bucket_maps,
            bucket_count,
            num_rows,
        }
    }

    /// Scan keys from a simd_json borrowed object (Pass 1)
    fn scan_keys<'a>(
        obj: &'a simd_json::borrowed::Object<'a>,
        prefix: &str,
        depth: usize,
        max_depth: usize,
        result: &mut SmallVec<[(String, InferredType); 32]>,
        key_buffer: &mut String,
    ) {
        use simd_json::BorrowedValue;

        for (key, value) in obj.iter() {
            key_buffer.clear();
            if !prefix.is_empty() {
                key_buffer.push_str(prefix);
                key_buffer.push('.');
            }
            key_buffer.push_str(key);

            match value {
                BorrowedValue::Object(nested) if depth < max_depth => {
                    let current_prefix = key_buffer.clone();
                    Self::scan_keys(nested, &current_prefix, depth + 1, max_depth, result, key_buffer);
                }
                _ => {
                    let inferred_type = Self::infer_type_borrowed(value);
                    result.push((key_buffer.clone(), inferred_type));
                }
            }
        }
    }

    /// Infer type from simd_json borrowed value
    #[inline]
    fn infer_type_borrowed(value: &simd_json::BorrowedValue<'_>) -> InferredType {
        use simd_json::BorrowedValue;
        match value {
            BorrowedValue::Static(simd_json::StaticNode::Null) => InferredType::Null,
            BorrowedValue::Static(simd_json::StaticNode::Bool(_)) => InferredType::Boolean,
            BorrowedValue::Static(simd_json::StaticNode::I64(_)) => InferredType::Int64,
            BorrowedValue::Static(simd_json::StaticNode::U64(u)) => {
                if *u > i64::MAX as u64 { InferredType::String } else { InferredType::Uint64 }
            }
            BorrowedValue::Static(simd_json::StaticNode::F64(_)) => InferredType::Float64,
            BorrowedValue::String(s) if s.is_empty() => InferredType::Null,
            BorrowedValue::String(_) => InferredType::String,
            BorrowedValue::Array(_) | BorrowedValue::Object(_) => InferredType::String,
        }
    }

    /// Extract values from a borrowed JSON object (Pass 2)
    #[allow(clippy::too_many_arguments)]
    fn extract_values<'a>(
        obj: &'a simd_json::borrowed::Object<'a>,
        prefix: &str,
        depth: usize,
        max_depth: usize,
        key_buffer: &mut String,
        key_to_idx: &AHashMap<String, usize>,
        key_to_builder: &[Option<usize>],
        subcolumn_indices: &AHashSet<usize>,
        overflow_indices: &AHashSet<usize>,
        keys: &[String],
        key_to_bucket: &[usize],
        result: &mut RowParseResult,
    ) {
        use simd_json::BorrowedValue;

        for (key, value) in obj.iter() {
            key_buffer.clear();
            if !prefix.is_empty() {
                key_buffer.push_str(prefix);
                key_buffer.push('.');
            }
            key_buffer.push_str(key);

            match value {
                BorrowedValue::Object(nested) if depth < max_depth => {
                    let current_prefix = key_buffer.clone();
                    Self::extract_values(
                        nested, &current_prefix, depth + 1, max_depth, key_buffer,
                        key_to_idx, key_to_builder, subcolumn_indices, overflow_indices,
                        keys, key_to_bucket, result,
                    );
                }
                _ => {
                    if let Some(&key_idx) = key_to_idx.get(key_buffer.as_str()) {
                        let compact_value = CompactValue::from_simd_borrowed(value);

                        if subcolumn_indices.contains(&key_idx) {
                            if let Some(builder_idx) = key_to_builder[key_idx] {
                                result.subcolumn_values.push((builder_idx, compact_value));
                            }
                        } else if overflow_indices.contains(&key_idx) && !compact_value.is_null_or_empty() {
                            let bucket_idx = key_to_bucket[key_idx];
                            // Store key_idx instead of cloning the key string
                            result.overflow_entries.push((bucket_idx, key_idx, compact_value));
                        }
                    }
                }
            }
        }
    }

    /// Hash a key to a bucket index using MurmurHash3 128-bit
    #[inline]
    fn hash_key_to_bucket(key: &str, bucket_count: usize) -> usize {
        use std::io::Cursor;
        let hash = murmur3::murmur3_x64_128(&mut Cursor::new(key.as_bytes()), 0)
            .expect("murmur3 hash should not fail");
        (hash as usize) % bucket_count
    }

    /// Get the final type for a key, checking type hints first
    fn get_type_for_key(&self, key: &str, inferred_type: InferredType) -> InferredType {
        if let Some(hints) = &self.config.type_hints {
            if let Some(hint) = hints.get(key) {
                return match hint {
                    JsonTypeHint::String => InferredType::String,
                    JsonTypeHint::Int64 => InferredType::Int64,
                    JsonTypeHint::Uint64 => InferredType::Uint64,
                    JsonTypeHint::Float64 => InferredType::Float64,
                    JsonTypeHint::Boolean => InferredType::Boolean,
                };
            }
        }
        inferred_type
    }
}

impl ProcessedJsonColumns {
    /// Convert subcolumns to Arrow arrays
    pub fn subcolumns_to_arrays(&self) -> Vec<(String, ArrayRef)> {
        self.subcolumns
            .iter()
            .map(|(name, data)| (name.clone(), data.to_arrow_array()))
            .collect()
    }

    /// Convert bucket maps to Arrow Map arrays
    pub fn bucket_maps_to_arrays(&self) -> Vec<(String, ArrayRef)> {
        (0..self.bucket_count)
            .map(|bucket_idx| {
                let name = format!("{}__json_type_bucket_{}", self.column_name, bucket_idx);
                let bucket_data = &self.bucket_maps[bucket_idx];

                let key_builder = LargeStringBuilder::new();
                let value_builder = LargeStringBuilder::new();
                let mut map_builder = MapBuilder::new(None, key_builder, value_builder);

                for row_idx in 0..self.num_rows {
                    if let Some(row_map) = bucket_data.get(&row_idx) {
                        let mut sorted_entries: Vec<_> = row_map.iter().collect();
                        sorted_entries.sort_by(|a, b| a.0.cmp(b.0));

                        for (k, v) in sorted_entries {
                            map_builder.keys().append_value(k);
                            map_builder.values().append_value(v);
                        }
                    }
                    map_builder.append(true).expect("map append should not fail");
                }

                let array: ArrayRef = Arc::new(map_builder.finish());
                (name, array)
            })
            .collect()
    }

    /// Convert original values to Arrow array
    pub fn original_to_array(&self) -> Option<(String, ArrayRef)> {
        self.original_values.as_ref().map(|values| {
            let mut builder = LargeStringBuilder::with_capacity(values.len(), values.len() * 100);
            for value in values {
                match value {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                }
            }
            let array: ArrayRef = Arc::new(builder.finish());
            (self.column_name.clone(), array)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> JsonColumnConfig {
        JsonColumnConfig {
            column: "properties".to_string(),
            max_subcolumns: 10,
            bucket_count: 4,
            max_depth: 3,
            keep_original_column: true,
            type_hints: None,
        }
    }

    #[test]
    fn test_flatten_simple_object() {
        let config = create_test_config();
        let processor = JsonColumnProcessor::new(config);

        let json_values = vec![
            Some(r#"{"name": "test", "value": 42}"#),
            Some(r#"{"name": "test2", "value": 100}"#),
        ];

        let result = processor.process_batch(json_values.into_iter());

        assert!(result.subcolumns.contains_key("properties.name"));
        assert!(result.subcolumns.contains_key("properties.value"));

        if let SubcolumnData::String(values) = &result.subcolumns["properties.name"] {
            assert_eq!(values.len(), 2);
            assert_eq!(values[0], Some("test".to_string()));
            assert_eq!(values[1], Some("test2".to_string()));
        } else {
            panic!("Expected string type for name");
        }
    }

    #[test]
    fn test_flatten_nested_object() {
        let config = create_test_config();
        let processor = JsonColumnProcessor::new(config);

        let json_values = vec![Some(r#"{"user": {"name": "Alice", "age": 30}}"#)];

        let result = processor.process_batch(json_values.into_iter());

        assert!(result.subcolumns.contains_key("properties.user.name"));
        assert!(result.subcolumns.contains_key("properties.user.age"));
    }

    #[test]
    fn test_type_inference() {
        let config = create_test_config();
        let processor = JsonColumnProcessor::new(config);

        let json_values = vec![Some(r#"{"count": 10, "rate": 3.14, "active": true, "name": "test"}"#)];

        let result = processor.process_batch(json_values.into_iter());

        assert!(
            matches!(result.subcolumns.get("properties.count"), Some(SubcolumnData::Uint64(_)) | Some(SubcolumnData::Int64(_))),
            "count should be numeric"
        );

        assert!(
            matches!(result.subcolumns.get("properties.rate"), Some(SubcolumnData::Float64(_))),
            "rate should be float"
        );

        assert!(
            matches!(result.subcolumns.get("properties.active"), Some(SubcolumnData::Boolean(_))),
            "active should be boolean"
        );

        assert!(
            matches!(result.subcolumns.get("properties.name"), Some(SubcolumnData::String(_))),
            "name should be string"
        );
    }

    #[test]
    fn test_overflow_to_buckets() {
        let mut config = create_test_config();
        config.max_subcolumns = 2;

        let processor = JsonColumnProcessor::new(config);

        let json_values = vec![Some(r#"{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}"#)];

        let result = processor.process_batch(json_values.into_iter());

        assert!(result.subcolumns.len() <= 2);

        let total_bucket_entries: usize = result
            .bucket_maps
            .iter()
            .flat_map(|bucket| bucket.values())
            .map(|map| map.len())
            .sum();

        assert!(total_bucket_entries >= 3);
    }

    #[test]
    fn test_hash_distribution() {
        let buckets: Vec<usize> = (0..100)
            .map(|i| JsonColumnProcessor::hash_key_to_bucket(&format!("key_{}", i), 4))
            .collect();

        let mut bucket_used = [false; 4];
        for b in buckets {
            bucket_used[b] = true;
        }
        assert!(bucket_used.iter().all(|&x| x), "All buckets should be used");
    }

    #[test]
    fn test_duplicate_keys_from_dotted_and_nested() {
        let config = create_test_config();
        let processor = JsonColumnProcessor::new(config);

        let json_values = vec![
            Some(r#"{"user.name": "literal_dotted", "user": {"name": "nested"}}"#),
            Some(r#"{"user": {"name": "only_nested"}}"#),
        ];

        let result = processor.process_batch(json_values.into_iter());

        if let Some(SubcolumnData::String(values)) = result.subcolumns.get("properties.user.name") {
            assert_eq!(values.len(), 2, "Should have exactly 2 rows");
            assert!(values[0].is_some(), "First row should have a value");
            assert_eq!(values[1], Some("only_nested".to_string()), "Second row should be 'only_nested'");
        } else {
            panic!("Expected string subcolumn for user.name");
        }
    }

    #[test]
    fn test_first_non_null_value_wins() {
        let config = create_test_config();
        let processor = JsonColumnProcessor::new(config);

        let json_values = vec![
            Some(r#"{"val.key": null, "val": {"key": "good_value"}}"#),
            Some(r#"{"val.key": "", "val": {"key": "also_good"}}"#),
        ];

        let result = processor.process_batch(json_values.into_iter());

        if let Some(SubcolumnData::String(values)) = result.subcolumns.get("properties.val.key") {
            assert_eq!(values.len(), 2, "Should have exactly 2 rows");
            assert_eq!(values[0], Some("good_value".to_string()), "First row: non-null should win");
            assert_eq!(values[1], Some("also_good".to_string()), "Second row: non-empty should win");
        } else {
            panic!("Expected string subcolumn for val.key");
        }
    }

    #[test]
    fn test_multiple_rows_alignment() {
        let config = create_test_config();
        let processor = JsonColumnProcessor::new(config);

        let json_values = vec![
            Some(r#"{"a": 1, "b": 2, "c": 3}"#),
            Some(r#"{"a": 10}"#),
            Some(r#"{"b": 20, "c": 30}"#),
            Some(r#"{"a": 100, "b": 200, "c": 300}"#),
        ];

        let result = processor.process_batch(json_values.into_iter());

        for (name, data) in &result.subcolumns {
            let len = match data {
                SubcolumnData::String(v) => v.len(),
                SubcolumnData::Int64(v) => v.len(),
                SubcolumnData::Uint64(v) => v.len(),
                SubcolumnData::Float64(v) => v.len(),
                SubcolumnData::Boolean(v) => v.len(),
            };
            assert_eq!(len, 4, "Subcolumn {} should have 4 rows, got {}", name, len);
        }
    }

    #[test]
    fn test_type_hints_override_inferred() {
        let mut type_hints = BTreeMap::new();
        type_hints.insert("count".to_string(), JsonTypeHint::String);
        type_hints.insert("id".to_string(), JsonTypeHint::Int64);

        let config = JsonColumnConfig {
            column: "properties".to_string(),
            max_subcolumns: 10,
            bucket_count: 4,
            max_depth: 3,
            keep_original_column: false,
            type_hints: Some(type_hints),
        };

        let processor = JsonColumnProcessor::new(config);

        let json_values = vec![
            Some(r#"{"count": 42, "id": "123", "name": "test"}"#),
            Some(r#"{"count": 100, "id": "456", "name": "test2"}"#),
        ];

        let result = processor.process_batch(json_values.into_iter());

        assert!(
            matches!(result.subcolumns.get("properties.count"), Some(SubcolumnData::String(_))),
            "count should be String due to type hint"
        );

        assert!(
            matches!(result.subcolumns.get("properties.id"), Some(SubcolumnData::Int64(_))),
            "id should be Int64 due to type hint"
        );

        assert!(
            matches!(result.subcolumns.get("properties.name"), Some(SubcolumnData::String(_))),
            "name should be String (inferred)"
        );

        if let Some(SubcolumnData::Int64(values)) = result.subcolumns.get("properties.id") {
            assert_eq!(values[0], Some(123));
            assert_eq!(values[1], Some(456));
        }
    }
}

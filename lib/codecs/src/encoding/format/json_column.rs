//! JSON column expansion for Parquet encoding
//!
//! This module provides functionality to expand JSON string columns into
//! individual subcolumns for efficient columnar storage, similar to ClickHouse's JSON type.

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use rayon::prelude::*;

use arrow::array::{
    ArrayRef, BooleanBuilder, Float64Builder, Int64Builder, LargeStringBuilder, MapBuilder,
    StringBuilder, UInt64Builder,
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
    /// Convert from simd_json owned value
    #[inline]
    fn from_simd_owned(value: simd_json::OwnedValue) -> Self {
        use simd_json::OwnedValue;
        match value {
            OwnedValue::Static(simd_json::StaticNode::Null) => CompactValue::Null,
            OwnedValue::Static(simd_json::StaticNode::Bool(b)) => CompactValue::Bool(b),
            OwnedValue::Static(simd_json::StaticNode::I64(i)) => CompactValue::Int(i),
            OwnedValue::Static(simd_json::StaticNode::U64(u)) => {
                if u > i64::MAX as u64 {
                    CompactValue::String(u.to_string())
                } else {
                    CompactValue::Uint(u)
                }
            }
            OwnedValue::Static(simd_json::StaticNode::F64(f)) => CompactValue::Float(f),
            OwnedValue::String(s) => CompactValue::String(s),
            OwnedValue::Array(_) | OwnedValue::Object(_) => {
                CompactValue::String(simd_json::to_string(&value).unwrap_or_default())
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
    fn infer_type(&self) -> InferredType {
        match self {
            CompactValue::Null => InferredType::Null,
            CompactValue::Bool(_) => InferredType::Boolean,
            CompactValue::Int(_) => InferredType::Int64,
            CompactValue::Uint(u) => {
                if *u > i64::MAX as u64 {
                    InferredType::String
                } else {
                    InferredType::Uint64
                }
            }
            CompactValue::Float(_) => InferredType::Float64,
            CompactValue::String(_) => InferredType::String,
        }
    }

    #[inline]
    fn to_string(&self) -> String {
        match self {
            CompactValue::Null => String::new(),
            CompactValue::Bool(b) => b.to_string(),
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
    /// Bucket maps for overflow keys (bucket_index -> (key -> value) pairs per row)
    pub bucket_maps: Vec<Vec<HashMap<String, String>>>,
    /// Number of buckets
    pub bucket_count: usize,
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
                let mut builder = StringBuilder::with_capacity(values.len(), values.len() * 32);
                for value in values {
                    match value {
                        Some(v) => builder.append_value(v),
                        None => builder.append_null(),
                    }
                }
                std::sync::Arc::new(builder.finish())
            }
            SubcolumnData::Int64(values) => {
                let mut builder = Int64Builder::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(v) => builder.append_value(*v),
                        None => builder.append_null(),
                    }
                }
                std::sync::Arc::new(builder.finish())
            }
            SubcolumnData::Uint64(values) => {
                let mut builder = UInt64Builder::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(v) => builder.append_value(*v),
                        None => builder.append_null(),
                    }
                }
                std::sync::Arc::new(builder.finish())
            }
            SubcolumnData::Float64(values) => {
                let mut builder = Float64Builder::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(v) => builder.append_value(*v),
                        None => builder.append_null(),
                    }
                }
                std::sync::Arc::new(builder.finish())
            }
            SubcolumnData::Boolean(values) => {
                let mut builder = BooleanBuilder::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(v) => builder.append_value(*v),
                        None => builder.append_null(),
                    }
                }
                std::sync::Arc::new(builder.finish())
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

/// Builder for subcolumn data that collects values efficiently
enum SubcolumnBuilderCompact {
    String(Vec<Option<String>>),
    Int64(Vec<Option<i64>>),
    Uint64(Vec<Option<u64>>),
    Float64(Vec<Option<f64>>),
    Boolean(Vec<Option<bool>>),
}

impl SubcolumnBuilderCompact {
    fn new(inferred_type: InferredType, capacity: usize) -> Self {
        match inferred_type {
            InferredType::String | InferredType::Null => {
                SubcolumnBuilderCompact::String(Vec::with_capacity(capacity))
            }
            InferredType::Int64 => SubcolumnBuilderCompact::Int64(Vec::with_capacity(capacity)),
            InferredType::Uint64 => SubcolumnBuilderCompact::Uint64(Vec::with_capacity(capacity)),
            InferredType::Float64 => SubcolumnBuilderCompact::Float64(Vec::with_capacity(capacity)),
            InferredType::Boolean => SubcolumnBuilderCompact::Boolean(Vec::with_capacity(capacity)),
        }
    }

    #[inline]
    fn push(&mut self, value: Option<&CompactValue>) {
        match self {
            SubcolumnBuilderCompact::String(vec) => {
                vec.push(value.map(|v| v.to_string()));
            }
            SubcolumnBuilderCompact::Int64(vec) => {
                vec.push(value.and_then(|v| v.as_i64()));
            }
            SubcolumnBuilderCompact::Uint64(vec) => {
                vec.push(value.and_then(|v| v.as_u64()));
            }
            SubcolumnBuilderCompact::Float64(vec) => {
                vec.push(value.and_then(|v| v.as_f64()));
            }
            SubcolumnBuilderCompact::Boolean(vec) => {
                vec.push(value.and_then(|v| v.as_bool()));
            }
        }
    }

    fn finish(self) -> SubcolumnData {
        match self {
            SubcolumnBuilderCompact::String(vec) => SubcolumnData::String(vec),
            SubcolumnBuilderCompact::Int64(vec) => SubcolumnData::Int64(vec),
            SubcolumnBuilderCompact::Uint64(vec) => SubcolumnData::Uint64(vec),
            SubcolumnBuilderCompact::Float64(vec) => SubcolumnData::Float64(vec),
            SubcolumnBuilderCompact::Boolean(vec) => SubcolumnData::Boolean(vec),
        }
    }
}

/// Statistics for a single key during processing
struct KeyStats {
    count: usize,
    inferred_type: InferredType,
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
    pub fn process_batch<'a, I>(&self, json_values: I) -> ProcessedJsonColumns
    where
        I: Iterator<Item = Option<&'a str>>,
    {
        let json_values: Vec<Option<&str>> = json_values.collect();
        let num_rows = json_values.len();

        // Phase 1: Parse and flatten JSON in parallel using simd-json
        let max_depth = self.config.max_depth;
        let keep_original = self.config.keep_original_column;

        let parse_results: Vec<(Option<String>, Vec<(String, CompactValue)>)> = json_values
            .par_iter()
            .map(|json_str| match json_str {
                Some(s) if !s.is_empty() => {
                    let original = if keep_original {
                        Some((*s).to_string())
                    } else {
                        None
                    };
                    let mut s_owned = s.to_string();
                    // SAFETY: simd_json::from_str requires mutable access
                    let parse_result =
                        unsafe { simd_json::from_str::<simd_json::OwnedValue>(&mut s_owned) };

                    match parse_result {
                        Ok(simd_json::OwnedValue::Object(obj)) => {
                            let mut flattened = Vec::with_capacity(obj.len() * 2);
                            let mut key_buffer = String::with_capacity(128);
                            Self::flatten_simd_to_strings(
                                *obj,
                                "",
                                0,
                                max_depth,
                                &mut flattened,
                                &mut key_buffer,
                            );
                            (original, flattened)
                        }
                        _ => (original, Vec::new()),
                    }
                }
                _ => (None, Vec::new())
            })
            .collect();

        // Phase 2: Sequential key interning and statistics gathering
        let mut key_interner: HashMap<Rc<str>, usize> = HashMap::with_capacity(256);
        let mut interned_keys: Vec<Rc<str>> = Vec::with_capacity(256);
        let mut key_stats: Vec<KeyStats> = Vec::with_capacity(256);
        let mut all_flattened: Vec<Vec<(usize, CompactValue)>> = Vec::with_capacity(num_rows);
        let mut original_values: Vec<Option<String>> = if self.config.keep_original_column {
            Vec::with_capacity(num_rows)
        } else {
            Vec::new()
        };

        for (original, flattened_strings) in parse_results {
            if self.config.keep_original_column {
                original_values.push(original);
            }

            let mut flattened: Vec<(usize, CompactValue)> =
                Vec::with_capacity(flattened_strings.len());
            for (key, value) in flattened_strings {
                let key_rc: Rc<str> = key.into();
                let key_idx = if let Some(&idx) = key_interner.get(&key_rc) {
                    idx
                } else {
                    let idx = interned_keys.len();
                    key_interner.insert(key_rc.clone(), idx);
                    interned_keys.push(key_rc);
                    key_stats.push(KeyStats {
                        count: 0,
                        inferred_type: InferredType::Null,
                    });
                    idx
                };

                if !value.is_null_or_empty() {
                    let stats = &mut key_stats[key_idx];
                    stats.count += 1;
                    stats.inferred_type = stats.inferred_type.resolve_conflict(value.infer_type());
                }

                flattened.push((key_idx, value));
            }
            all_flattened.push(flattened);
        }

        // Phase 3: Sort keys by frequency and determine subcolumns vs overflow
        let mut sorted_key_indices: Vec<usize> = (0..interned_keys.len()).collect();
        sorted_key_indices.sort_unstable_by(|&a, &b| {
            key_stats[b]
                .count
                .cmp(&key_stats[a].count)
                .then(interned_keys[a].cmp(&interned_keys[b]))
        });

        let overflow_key_indices: std::collections::HashSet<usize> = sorted_key_indices
            .iter()
            .skip(self.config.max_subcolumns)
            .copied()
            .collect();

        // Phase 4: Build subcolumns and bucket maps
        let mut subcolumn_builders: Vec<(usize, SubcolumnBuilderCompact)> = sorted_key_indices
            .iter()
            .take(self.config.max_subcolumns)
            .map(|&idx| {
                let key = &interned_keys[idx];
                let final_type = self.get_type_for_key(key, key_stats[idx].inferred_type);
                (idx, SubcolumnBuilderCompact::new(final_type, num_rows))
            })
            .collect();

        let mut bucket_maps: Vec<Vec<Option<HashMap<String, String>>>> =
            vec![vec![None; num_rows]; self.config.bucket_count];

        let num_keys = interned_keys.len();
        let mut key_to_builder: Vec<Option<usize>> = vec![None; num_keys];
        for (builder_idx, (key_idx, _)) in subcolumn_builders.iter().enumerate() {
            key_to_builder[*key_idx] = Some(builder_idx);
        }

        let num_builders = subcolumn_builders.len();
        let mut row_values: Vec<Option<&CompactValue>> = vec![None; num_builders];
        let mut row_has_good_value: Vec<bool> = vec![false; num_builders];

        for (row_idx, flattened) in all_flattened.iter().enumerate() {
            row_values.iter_mut().for_each(|x| *x = None);
            row_has_good_value.iter_mut().for_each(|x| *x = false);

            for (key_idx, value) in flattened {
                if let Some(builder_idx) = key_to_builder.get(*key_idx).copied().flatten() {
                    // First non-null, non-empty value wins for subcolumns
                    let is_good_value = !value.is_null_or_empty();
                    if row_values[builder_idx].is_none()
                        || (!row_has_good_value[builder_idx] && is_good_value)
                    {
                        row_values[builder_idx] = Some(value);
                        row_has_good_value[builder_idx] = is_good_value;
                    }
                } else if overflow_key_indices.contains(key_idx) && !value.is_null_or_empty() {
                    let key = &interned_keys[*key_idx];
                    let bucket_idx = self.hash_key_to_bucket(key);
                    let value_str = value.to_string();
                    bucket_maps[bucket_idx][row_idx]
                        .get_or_insert_with(HashMap::new)
                        .insert(key.to_string(), value_str);
                }
            }

            for builder_idx in 0..num_builders {
                subcolumn_builders[builder_idx].1.push(row_values[builder_idx]);
            }
        }

        let mut subcolumns: BTreeMap<String, SubcolumnData> = BTreeMap::new();
        for (key_idx, builder) in subcolumn_builders {
            let key = &interned_keys[key_idx];
            let full_key = format!("{}.{}", self.config.column, key);
            subcolumns.insert(full_key, builder.finish());
        }

        drop(all_flattened);

        let bucket_maps: Vec<Vec<HashMap<String, String>>> = bucket_maps
            .into_iter()
            .map(|bucket| {
                bucket
                    .into_iter()
                    .map(|opt_hm| opt_hm.unwrap_or_default())
                    .collect()
            })
            .collect();

        ProcessedJsonColumns {
            column_name: self.config.column.clone(),
            original_values: if self.config.keep_original_column {
                Some(original_values)
            } else {
                None
            },
            subcolumns,
            bucket_maps,
            bucket_count: self.config.bucket_count,
        }
    }

    /// Flatten a simd_json object into String keys and CompactValues
    fn flatten_simd_to_strings(
        obj: simd_json::owned::Object,
        prefix: &str,
        depth: usize,
        max_depth: usize,
        result: &mut Vec<(String, CompactValue)>,
        key_buffer: &mut String,
    ) {
        use simd_json::OwnedValue;

        for (key, value) in obj {
            key_buffer.clear();
            if !prefix.is_empty() {
                key_buffer.push_str(prefix);
                key_buffer.push('.');
            }
            key_buffer.push_str(&key);

            match value {
                OwnedValue::Object(nested) if depth < max_depth => {
                    let current_prefix = key_buffer.clone();
                    Self::flatten_simd_to_strings(
                        *nested,
                        &current_prefix,
                        depth + 1,
                        max_depth,
                        result,
                        key_buffer,
                    );
                }
                _ => {
                    result.push((key_buffer.clone(), CompactValue::from_simd_owned(value)));
                }
            }
        }
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

    /// Hash a key to a bucket index using MurmurHash3 128-bit
    fn hash_key_to_bucket(&self, key: &str) -> usize {
        use std::io::Cursor;
        let hash = murmur3::murmur3_x64_128(&mut Cursor::new(key.as_bytes()), 0)
            .expect("murmur3 hash should not fail on in-memory data");
        (hash as usize) % self.config.bucket_count
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

                // Use LargeStringBuilder (i64 offsets) to avoid overflow with large batches
                let key_builder = LargeStringBuilder::new();
                let value_builder = LargeStringBuilder::new();
                let mut map_builder = MapBuilder::new(None, key_builder, value_builder);

                for row_map in bucket_data {
                    let mut sorted_entries: Vec<_> = row_map.iter().collect();
                    sorted_entries.sort_by(|a, b| a.0.cmp(b.0));

                    for (k, v) in sorted_entries {
                        map_builder.keys().append_value(k);
                        map_builder.values().append_value(v);
                    }
                    map_builder.append(true).unwrap();
                }

                let array: ArrayRef = std::sync::Arc::new(map_builder.finish());
                (name, array)
            })
            .collect()
    }

    /// Convert original values to Arrow array
    pub fn original_to_array(&self) -> Option<(String, ArrayRef)> {
        self.original_values.as_ref().map(|values| {
            let mut builder = StringBuilder::with_capacity(values.len(), values.len() * 100);
            for value in values {
                match value {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                }
            }
            let array: ArrayRef = std::sync::Arc::new(builder.finish());
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

        let json_values =
            vec![Some(r#"{"count": 10, "rate": 3.14, "active": true, "name": "test"}"#)];

        let result = processor.process_batch(json_values.into_iter());

        assert!(
            matches!(
                result.subcolumns.get("properties.count"),
                Some(SubcolumnData::Uint64(_)) | Some(SubcolumnData::Int64(_))
            ),
            "count should be numeric"
        );

        assert!(
            matches!(
                result.subcolumns.get("properties.rate"),
                Some(SubcolumnData::Float64(_))
            ),
            "rate should be float"
        );

        assert!(
            matches!(
                result.subcolumns.get("properties.active"),
                Some(SubcolumnData::Boolean(_))
            ),
            "active should be boolean"
        );

        assert!(
            matches!(
                result.subcolumns.get("properties.name"),
                Some(SubcolumnData::String(_))
            ),
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
            .flat_map(|bucket| bucket.iter())
            .map(|map| map.len())
            .sum();

        assert!(total_bucket_entries >= 3);
    }

    #[test]
    fn test_hash_distribution() {
        let config = create_test_config();
        let processor = JsonColumnProcessor::new(config);

        let buckets: Vec<usize> = (0..100)
            .map(|i| processor.hash_key_to_bucket(&format!("key_{}", i)))
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
            assert_eq!(
                values[1],
                Some("only_nested".to_string()),
                "Second row should be 'only_nested'"
            );
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
            assert_eq!(
                values[0],
                Some("good_value".to_string()),
                "First row: non-null should win over null"
            );
            assert_eq!(
                values[1],
                Some("also_good".to_string()),
                "Second row: non-empty should win over empty"
            );
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
            matches!(
                result.subcolumns.get("properties.count"),
                Some(SubcolumnData::String(_))
            ),
            "count should be String due to type hint"
        );

        assert!(
            matches!(
                result.subcolumns.get("properties.id"),
                Some(SubcolumnData::Int64(_))
            ),
            "id should be Int64 due to type hint"
        );

        assert!(
            matches!(
                result.subcolumns.get("properties.name"),
                Some(SubcolumnData::String(_))
            ),
            "name should be String (inferred)"
        );

        if let Some(SubcolumnData::Int64(values)) = result.subcolumns.get("properties.id") {
            assert_eq!(values[0], Some(123));
            assert_eq!(values[1], Some(456));
        }
    }
}

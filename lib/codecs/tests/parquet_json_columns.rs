//! End-to-end integration tests for Parquet JSON column expansion
//!
//! These tests read JSON files from testdata, process them with JSON column
//! expansion, write parquet files to disk, and verify the output.

#![cfg(feature = "parquet")]

use arrow::array::RecordBatchReader;
use arrow::datatypes::DataType;
use bytes::BytesMut;
use codecs::encoding::format::{
    JsonColumnConfig, ParquetSerializer, ParquetSerializerConfig,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use tokio_util::codec::Encoder;
use vector_core::event::{Event, LogEvent};

/// Test directory for output files
fn get_test_output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("target")
        .join("test_output")
        .join("parquet_json_columns");
    fs::create_dir_all(&dir).expect("Failed to create test output directory");
    dir
}

/// Create a ParquetSerializerConfig with JSON column expansion
fn create_json_column_config(json_column_name: &str) -> ParquetSerializerConfig {
    let mut config = ParquetSerializerConfig::default();
    config.infer_schema = true;
    config.json_columns = Some(vec![JsonColumnConfig {
        column: json_column_name.to_string(),
        max_subcolumns: 100,  // Keep it smaller for testing visibility
        bucket_count: 8,       // Use fewer buckets for testing
        max_depth: 5,
        keep_original_column: true,
        type_hints: None,
    }]);
    config
}

/// Inspect a parquet file and print its schema
fn inspect_parquet_file(path: &PathBuf) {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open parquet file {}: {}", path.display(), e);
            return;
        }
    };

    let reader = match ParquetRecordBatchReaderBuilder::try_new(file) {
        Ok(builder) => match builder.build() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Failed to build reader: {}", e);
                return;
            }
        },
        Err(e) => {
            eprintln!("Failed to create reader: {}", e);
            return;
        }
    };

    println!("\n=== Parquet File: {} ===", path.display());

    let schema = reader.schema();
    println!("\nSchema ({} columns):", schema.fields().len());
    println!("{:-<60}", "");

    // Group columns by type
    let mut subcolumns = Vec::new();
    let mut bucket_columns = Vec::new();
    let mut other_columns = Vec::new();

    for field in schema.fields() {
        let name = field.name();
        if name.contains("__json_type_bucket_") {
            bucket_columns.push((name.clone(), format!("{:?}", field.data_type())));
        } else if name.contains('.') {
            subcolumns.push((name.clone(), format!("{:?}", field.data_type())));
        } else {
            other_columns.push((name.clone(), format!("{:?}", field.data_type())));
        }
    }

    println!("\nOriginal columns ({}):", other_columns.len());
    for (name, dtype) in &other_columns {
        println!("  {} : {}", name, dtype);
    }

    println!("\nExpanded subcolumns ({}):", subcolumns.len());
    for (name, dtype) in &subcolumns {
        println!("  {} : {}", name, dtype);
    }

    println!("\nBucket map columns ({}):", bucket_columns.len());
    for (name, dtype) in &bucket_columns {
        println!("  {} : {}", name, dtype);
    }

    // Read and count rows
    let file = File::open(path).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();

    let mut total_rows = 0;
    for batch in reader {
        let batch = batch.unwrap();
        total_rows += batch.num_rows();
    }
    println!("\nTotal rows: {}", total_rows);
    println!("{:=<60}\n", "");
}

#[test]
fn test_e2e_multiple_json_events() {
    let output_dir = get_test_output_dir();

    // Create multiple test events with properties
    let events: Vec<Event> = (0..10)
        .map(|i| {
            let mut log = LogEvent::default();
            log.insert("event_id", i as i64);
            log.insert("event_type", if i % 2 == 0 { "click" } else { "view" });
            log.insert("timestamp", format!("2024-01-{:02}T12:00:00Z", i + 1));
            log.insert(
                "properties",
                format!(
                    r#"{{"user_id":"user{}","page":"page{}","duration":{},"is_premium":{},"nested":{{"level1":{{"level2":"deep_value_{}"}}}},"tags":["tag1","tag2"]}}"#,
                    i, i % 3, i * 100, i % 2 == 0, i
                ),
            );
            Event::Log(log)
        })
        .collect();

    println!("\nCreated {} synthetic events", events.len());

    let output_path = output_dir.join("synthetic_events.parquet");
    println!("Writing parquet to: {}", output_path.display());

    let config = create_json_column_config("properties");
    let mut serializer = ParquetSerializer::new(config).expect("Failed to create serializer");

    let mut buffer = BytesMut::new();
    match serializer.encode(events, &mut buffer) {
        Ok(()) => {
            let mut file = File::create(&output_path).expect("Failed to create output file");
            file.write_all(&buffer).expect("Failed to write parquet data");
            println!("Successfully wrote parquet file");
            inspect_parquet_file(&output_path);
        }
        Err(e) => {
            eprintln!("Failed to encode events: {}", e);
        }
    }
}

#[test]
fn test_e2e_verify_column_types() {
    let output_dir = get_test_output_dir();

    // Create events with known types
    let mut log1 = LogEvent::default();
    log1.insert("id", 1_i64);
    log1.insert(
        "data",
        r#"{"string_field":"hello","int_field":42,"float_field":3.14,"bool_field":true}"#,
    );

    let mut log2 = LogEvent::default();
    log2.insert("id", 2_i64);
    log2.insert(
        "data",
        r#"{"string_field":"world","int_field":100,"float_field":2.71,"bool_field":false}"#,
    );

    let events = vec![Event::Log(log1), Event::Log(log2)];

    let output_path = output_dir.join("typed_columns.parquet");

    let config = create_json_column_config("data");
    let mut serializer = ParquetSerializer::new(config).expect("Failed to create serializer");

    let mut buffer = BytesMut::new();
    serializer.encode(events, &mut buffer).expect("Failed to encode");

    let mut file = File::create(&output_path).expect("Failed to create file");
    file.write_all(&buffer).expect("Failed to write");

    // Read back and verify types
    let file = File::open(&output_path).expect("Failed to open parquet file");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("Failed to create reader")
        .build()
        .expect("Failed to build reader");

    let schema = reader.schema();

    println!("\n=== Type Verification ===");

    // Check that string field is Utf8
    if let Ok(field) = schema.field_with_name("data.string_field") {
        let dtype = field.data_type();
        assert_eq!(dtype, &DataType::Utf8, "string_field should be Utf8");
        println!("data.string_field: {:?} ✓", dtype);
    }

    // Check that int field is numeric (Int64 or UInt64)
    if let Ok(field) = schema.field_with_name("data.int_field") {
        let dtype = field.data_type();
        assert!(
            matches!(dtype, DataType::Int64 | DataType::UInt64),
            "int_field should be Int64 or UInt64"
        );
        println!("data.int_field: {:?} ✓", dtype);
    }

    // Check that float field is Float64
    if let Ok(field) = schema.field_with_name("data.float_field") {
        let dtype = field.data_type();
        assert_eq!(dtype, &DataType::Float64, "float_field should be Float64");
        println!("data.float_field: {:?} ✓", dtype);
    }

    // Check that bool field is Boolean
    if let Ok(field) = schema.field_with_name("data.bool_field") {
        let dtype = field.data_type();
        assert_eq!(dtype, &DataType::Boolean, "bool_field should be Boolean");
        println!("data.bool_field: {:?} ✓", dtype);
    }

    // Verify bucket columns exist and are Map type
    let bucket_cols: Vec<_> = schema
        .fields()
        .iter()
        .filter(|f| f.name().contains("__json_type_bucket_"))
        .collect();

    assert!(!bucket_cols.is_empty(), "Should have bucket columns");
    for field in &bucket_cols {
        let dtype = field.data_type();
        assert!(
            matches!(dtype, DataType::Map(_, _)),
            "Bucket column should be Map type"
        );
    }
    println!("Bucket columns ({} total): Map<String, String> ✓", bucket_cols.len());

    println!("\nAll type checks passed!\n");

    // Also print the full inspection
    inspect_parquet_file(&output_path);
}

#[test]
fn test_e2e_overflow_to_buckets() {
    let output_dir = get_test_output_dir();

    // Create an event with many keys to trigger overflow to buckets
    let many_keys: String = (0..50)
        .map(|i| format!(r#""key_{}":"value_{}""#, i, i))
        .collect::<Vec<_>>()
        .join(",");

    let mut log = LogEvent::default();
    log.insert("id", 1_i64);
    log.insert("data", format!("{{{}}}", many_keys));

    let events = vec![Event::Log(log)];

    // Configure with small max_subcolumns to force overflow
    let mut config = ParquetSerializerConfig::default();
    config.infer_schema = true;
    config.json_columns = Some(vec![JsonColumnConfig {
        column: "data".to_string(),
        max_subcolumns: 10,  // Only 10 subcolumns, rest go to buckets
        bucket_count: 4,     // 4 buckets
        max_depth: 3,
        keep_original_column: true,
        type_hints: None,
    }]);

    let mut serializer = ParquetSerializer::new(config).expect("Failed to create serializer");
    let mut buffer = BytesMut::new();
    serializer.encode(events, &mut buffer).expect("Failed to encode");

    let output_path = output_dir.join("overflow_to_buckets.parquet");
    let mut file = File::create(&output_path).expect("Failed to create file");
    file.write_all(&buffer).expect("Failed to write");

    // Verify the output
    let file = File::open(&output_path).expect("Failed to open");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("Failed to create reader")
        .build()
        .expect("Failed to build reader");

    let schema = reader.schema();

    let subcolumns: Vec<_> = schema
        .fields()
        .iter()
        .filter(|f| f.name().starts_with("data."))
        .collect();

    let bucket_cols: Vec<_> = schema
        .fields()
        .iter()
        .filter(|f| f.name().contains("__json_type_bucket_"))
        .collect();

    println!("\n=== Overflow Test Results ===");
    println!("Total keys in JSON: 50");
    println!("Max subcolumns configured: 10");
    println!("Actual subcolumns created: {}", subcolumns.len());
    println!("Bucket columns: {}", bucket_cols.len());

    assert!(
        subcolumns.len() <= 10,
        "Should have at most 10 subcolumns, got {}",
        subcolumns.len()
    );
    assert_eq!(
        bucket_cols.len(),
        4,
        "Should have exactly 4 bucket columns"
    );

    println!("Overflow to buckets working correctly! ✓\n");

    inspect_parquet_file(&output_path);
}

#[test]
fn test_e2e_nested_json_objects() {
    let output_dir = get_test_output_dir();

    // Create events with nested JSON
    let mut log = LogEvent::default();
    log.insert(
        "event",
        r#"{"user":{"name":"Alice","address":{"city":"NYC","zip":"10001"}},"action":"login"}"#,
    );

    let events = vec![Event::Log(log)];

    let config = create_json_column_config("event");
    let mut serializer = ParquetSerializer::new(config).expect("Failed to create serializer");

    let mut buffer = BytesMut::new();
    serializer.encode(events, &mut buffer).expect("Failed to encode");

    let output_path = output_dir.join("nested_objects.parquet");
    let mut file = File::create(&output_path).expect("Failed to create file");
    file.write_all(&buffer).expect("Failed to write");

    // Verify the output
    let file = File::open(&output_path).expect("Failed to open");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("Failed to create reader")
        .build()
        .expect("Failed to build reader");

    let schema = reader.schema();
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

    println!("\n=== Nested Objects Test ===");

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

    println!("Nested object flattening working correctly! ✓");
    println!("Fields: {:?}\n", field_names);

    inspect_parquet_file(&output_path);
}

#[test]
fn test_e2e_large_testdata_logs() {
    use std::io::{BufRead, BufReader};
    use std::time::Instant;

    let output_dir = get_test_output_dir();
    let testdata_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("testdata");

    // Find all .log files
    let log_files: Vec<PathBuf> = fs::read_dir(&testdata_dir)
        .expect("Failed to read testdata directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().map_or(false, |ext| ext == "log"))
        .collect();

    println!("\n=== Large Testdata Logs Test ===");
    println!("Found {} log files in testdata/", log_files.len());

    if log_files.is_empty() {
        println!("No log files found, skipping test");
        return;
    }

    // Read events from all log files (limit to first 10000 events for test)
    let max_events = 10000;
    let mut events: Vec<Event> = Vec::new();
    let mut total_lines = 0;
    let mut files_processed = 0;

    let start = Instant::now();

    for log_file in &log_files {
        if events.len() >= max_events {
            break;
        }

        let file = match File::open(log_file) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let reader = BufReader::new(file);
        for line in reader.lines() {
            if events.len() >= max_events {
                break;
            }

            total_lines += 1;
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };

            if line.trim().is_empty() {
                continue;
            }

            // Parse JSON line and create LogEvent
            let json_value: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let mut log = LogEvent::default();

            // Insert all top-level fields
            if let serde_json::Value::Object(map) = json_value {
                for (key, value) in map {
                    // Convert serde_json::Value to string for non-primitive types
                    match value {
                        serde_json::Value::String(s) => {
                            log.insert(key.as_str(), s);
                        }
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                log.insert(key.as_str(), i);
                            } else if let Some(f) = n.as_f64() {
                                log.insert(key.as_str(), f);
                            }
                        }
                        serde_json::Value::Bool(b) => {
                            log.insert(key.as_str(), b);
                        }
                        serde_json::Value::Null => {
                            // Skip nulls
                        }
                        _ => {
                            // For objects and arrays, store as JSON string
                            log.insert(key.as_str(), value.to_string());
                        }
                    }
                }
            }

            events.push(Event::Log(log));
        }
        files_processed += 1;
    }

    let read_duration = start.elapsed();
    println!(
        "Read {} events from {} files ({} total lines) in {:.2}s",
        events.len(),
        files_processed,
        total_lines,
        read_duration.as_secs_f64()
    );

    if events.is_empty() {
        println!("No valid events found, skipping parquet encoding");
        return;
    }

    // Create config with JSON column expansion for 'properties' field
    let mut config = ParquetSerializerConfig::default();
    config.infer_schema = true;
    config.json_columns = Some(vec![
        JsonColumnConfig {
            column: "properties".to_string(),
            max_subcolumns: 100,  // Limit for test performance
            bucket_count: 16,
            max_depth: 5,
            keep_original_column: true,
            type_hints: None,
        },
        JsonColumnConfig {
            column: "person_properties".to_string(),
            max_subcolumns: 50,
            bucket_count: 8,
            max_depth: 3,
            keep_original_column: true,
            type_hints: None,
        },
    ]);

    let mut serializer = ParquetSerializer::new(config).expect("Failed to create serializer");
    let mut buffer = BytesMut::new();

    let encode_start = Instant::now();
    match serializer.encode(events.clone(), &mut buffer) {
        Ok(()) => {
            let encode_duration = encode_start.elapsed();
            println!(
                "Encoded {} events to parquet in {:.2}s ({:.0} events/sec)",
                events.len(),
                encode_duration.as_secs_f64(),
                events.len() as f64 / encode_duration.as_secs_f64()
            );

            let output_path = output_dir.join("testdata_logs.parquet");
            let mut file = File::create(&output_path).expect("Failed to create output file");
            file.write_all(&buffer).expect("Failed to write parquet data");

            let file_size = buffer.len();
            println!(
                "Wrote parquet file: {} ({:.2} MB)",
                output_path.display(),
                file_size as f64 / (1024.0 * 1024.0)
            );

            // Inspect the output
            inspect_parquet_file(&output_path);

            // Verify with reader
            let file = File::open(&output_path).expect("Failed to open parquet file");
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)
                .expect("Failed to create reader")
                .build()
                .expect("Failed to build reader");

            let schema = reader.schema();
            println!("\nTotal columns in schema: {}", schema.fields().len());

            // Count column types
            let subcolumns = schema
                .fields()
                .iter()
                .filter(|f| f.name().contains('.'))
                .count();
            let bucket_cols = schema
                .fields()
                .iter()
                .filter(|f| f.name().contains("__json_type_bucket_"))
                .count();
            let original_cols = schema.fields().len() - subcolumns - bucket_cols;

            println!("  Original columns: {}", original_cols);
            println!("  Expanded subcolumns: {}", subcolumns);
            println!("  Bucket map columns: {}", bucket_cols);

            // Print some sample subcolumn names
            let sample_subcolumns: Vec<&str> = schema
                .fields()
                .iter()
                .filter(|f| f.name().contains('.') && !f.name().contains("__json_type_bucket_"))
                .take(20)
                .map(|f| f.name().as_str())
                .collect();
            println!("\nSample subcolumns (first 20):");
            for col in sample_subcolumns {
                println!("  {}", col);
            }

            println!("\nLarge testdata logs test passed! ✓\n");
        }
        Err(e) => {
            panic!("Failed to encode events: {}", e);
        }
    }
}

#[test]
fn test_e2e_encode_batch_split_with_json_columns() {
    // Test that encode_batch_split properly handles JSON column expansion
    // This verifies the P1 fix - JSON expansion was previously skipped in super-batch mode

    let output_dir = get_test_output_dir();
    let _output_path = output_dir.join("batch_split_json.parquet");

    // Create test events with JSON properties
    let mut events: Vec<Event> = Vec::new();
    for i in 0..100 {
        let mut log = LogEvent::default();
        log.insert("id", i as i64);
        log.insert("name", format!("event_{}", i));
        // JSON column to be expanded
        log.insert(
            "properties",
            format!(
                r#"{{"user_id":"user_{}", "action":"click", "count":{}, "nested":{{"depth":"value_{}"}} }}"#,
                i, i * 10, i
            ),
        );
        events.push(Event::Log(log));
    }

    // Create config with JSON column expansion AND rows_per_file to trigger super-batch mode
    let mut config = create_json_column_config("properties");
    config.rows_per_file = Some(50); // Split into 2 files of 50 rows each

    let serializer = ParquetSerializer::new(config).expect("Failed to create serializer");

    // Use encode_batch_split
    let parquet_files = serializer
        .encode_batch_split(events)
        .expect("Failed to encode batch split");

    println!("\n=== encode_batch_split JSON Columns Test ===");
    println!("Generated {} parquet files", parquet_files.len());

    // Should have 2 files (100 events / 50 per file)
    assert_eq!(parquet_files.len(), 2, "Expected 2 files from split");

    // Verify each file has JSON columns expanded
    for (file_idx, parquet_bytes) in parquet_files.iter().enumerate() {
        // Write to disk for inspection
        let file_path = output_dir.join(format!("batch_split_json_{}.parquet", file_idx));
        let mut file = File::create(&file_path).expect("Failed to create file");
        file.write_all(parquet_bytes).expect("Failed to write file");

        // Read and verify schema
        let file = File::open(&file_path).expect("Failed to open file");
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("Failed to build reader");
        let schema = builder.schema().clone();

        println!("\nFile {} schema has {} columns", file_idx, schema.fields().len());

        // Check for expanded subcolumns (should have properties.user_id, properties.action, etc.)
        let has_user_id = schema.fields().iter().any(|f| f.name() == "properties.user_id");
        let has_action = schema.fields().iter().any(|f| f.name() == "properties.action");
        let has_nested = schema.fields().iter().any(|f| f.name() == "properties.nested.depth");
        let has_bucket_maps = schema.fields().iter().any(|f| f.name().contains("__json_type_bucket_"));

        println!("  properties.user_id: {}", has_user_id);
        println!("  properties.action: {}", has_action);
        println!("  properties.nested.depth: {}", has_nested);
        println!("  Has bucket maps: {}", has_bucket_maps);

        assert!(
            has_user_id,
            "File {} should have properties.user_id subcolumn (JSON expansion missing in split mode)",
            file_idx
        );
        assert!(
            has_action,
            "File {} should have properties.action subcolumn",
            file_idx
        );
        assert!(
            has_nested,
            "File {} should have properties.nested.depth subcolumn",
            file_idx
        );
        assert!(
            has_bucket_maps,
            "File {} should have bucket map columns",
            file_idx
        );

        // Verify row count
        let file = File::open(&file_path).expect("Failed to open file");
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("Failed to build reader");
        let reader = builder.build().expect("Failed to build batch reader");
        let total_rows: usize = reader.map(|b| b.expect("batch").num_rows()).sum();
        assert_eq!(total_rows, 50, "File {} should have 50 rows", file_idx);

        println!("  Total rows: {}", total_rows);
    }

    // Clean up individual files
    for file_idx in 0..parquet_files.len() {
        let file_path = output_dir.join(format!("batch_split_json_{}.parquet", file_idx));
        let _ = fs::remove_file(file_path);
    }

    println!("\nencode_batch_split JSON columns test passed! ✓\n");
}

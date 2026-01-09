//! Benchmark tests for Parquet encoding with JSON column expansion
//!
//! Run with: cargo test -p codecs --features parquet --release -- benchmark --nocapture --ignored

use bytes::{Bytes, BytesMut};
use codecs::encoding::format::{
    JsonColumnConfig, ParquetSerializer, ParquetSerializerConfig, SortingColumnConfig,
};
use std::time::Instant;
use tokio_util::codec::Encoder;
use vector_core::event::{Event, LogEvent};

fn create_test_events(count: usize) -> Vec<Event> {
    (0..count)
        .map(|i| {
            let mut log = LogEvent::default();
            log.insert("id", i as i64);
            log.insert("timestamp", format!("2026-01-01T00:00:{:02}.000Z", i % 60));
            log.insert(
                "properties",
                format!(
                    r#"{{"user_id":"user{}","session_id":"sess{}","page_views":{},"is_premium":{},"score":{},"tags":["tag1","tag2","tag3"],"metadata":{{"source":"web","version":"1.0","nested":{{"deep":"value{}"}}}},"extra_field_{}":"value{}"}}"#,
                    i,
                    i * 2,
                    i * 10,
                    i % 2 == 0,
                    i as f64 * 1.5,
                    i,
                    i % 100,
                    i
                ),
            );
            Event::Log(log)
        })
        .collect()
}

#[test]
#[ignore] // Run with --ignored flag
fn benchmark_json_column_expansion() {
    let event_counts = [1000, 5000, 10000];

    println!("\n=== JSON Column Expansion Benchmark ===\n");
    println!(
        "{:>10} {:>12} {:>12} {:>15}",
        "Events", "Time (ms)", "Throughput", "Output Size"
    );
    println!("{}", "-".repeat(55));

    for &count in &event_counts {
        let events = create_test_events(count);

        let mut config = ParquetSerializerConfig::default();
        config.infer_schema = true;
        config.json_columns = Some(vec![JsonColumnConfig {
            column: "properties".to_string(),
            max_subcolumns: 50,
            bucket_count: 8,
            max_depth: 5,
            keep_original_column: true,
            type_hints: None,
        }]);

        let mut serializer = ParquetSerializer::new(config).expect("Failed to create serializer");
        let mut buffer = BytesMut::new();

        let start = Instant::now();
        serializer
            .encode(events, &mut buffer)
            .expect("Failed to encode");
        let elapsed = start.elapsed();

        let throughput = count as f64 / elapsed.as_secs_f64();
        println!(
            "{:>10} {:>12.2} {:>12.0}/s {:>12} KB",
            count,
            elapsed.as_millis(),
            throughput,
            buffer.len() / 1024
        );
    }
}

#[test]
#[ignore]
fn benchmark_super_batch_with_json_expansion() {
    let total_events = 10000;
    let rows_per_file_options = [1000, 2500, 5000];

    println!("\n=== Super-Batch + JSON Expansion Benchmark ===\n");
    println!(
        "{:>10} {:>12} {:>12} {:>10} {:>12}",
        "Events", "Rows/File", "Time (ms)", "Files", "Total Size"
    );
    println!("{}", "-".repeat(65));

    for &rows_per_file in &rows_per_file_options {
        let events = create_test_events(total_events);

        let mut config = ParquetSerializerConfig::default();
        config.infer_schema = true;
        config.rows_per_file = Some(rows_per_file);
        config.json_columns = Some(vec![JsonColumnConfig {
            column: "properties".to_string(),
            max_subcolumns: 50,
            bucket_count: 8,
            max_depth: 5,
            keep_original_column: false,
            type_hints: None,
        }]);

        let serializer = ParquetSerializer::new(config).expect("Failed to create serializer");

        let start = Instant::now();
        let files = serializer
            .encode_batch_split(events)
            .expect("Failed to encode");
        let elapsed = start.elapsed();

        let total_size: usize = files.iter().map(|f: &Bytes| f.len()).sum();
        println!(
            "{:>10} {:>12} {:>12.2} {:>10} {:>12} KB",
            total_events,
            rows_per_file,
            elapsed.as_millis(),
            files.len(),
            total_size / 1024
        );
    }
}

#[test]
#[ignore]
fn benchmark_super_batch_with_sorting_and_json() {
    let total_events = 10000;

    println!("\n=== Super-Batch + Sorting + JSON Expansion Benchmark ===\n");
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "Events", "Rows/File", "Time (ms)", "Files"
    );
    println!("{}", "-".repeat(50));

    let events = create_test_events(total_events);

    let mut config = ParquetSerializerConfig::default();
    config.infer_schema = true;
    config.rows_per_file = Some(2000);
    config.sorting_columns = Some(vec![SortingColumnConfig {
        column: "id".to_string(),
        descending: false,
    }]);
    config.json_columns = Some(vec![JsonColumnConfig {
        column: "properties".to_string(),
        max_subcolumns: 50,
        bucket_count: 8,
        max_depth: 5,
        keep_original_column: false,
        type_hints: None,
    }]);

    let serializer = ParquetSerializer::new(config).expect("Failed to create serializer");

    let start = Instant::now();
    let files = serializer
        .encode_batch_split(events)
        .expect("Failed to encode");
    let elapsed = start.elapsed();

    println!(
        "{:>10} {:>12} {:>12.2} {:>10}",
        total_events,
        2000,
        elapsed.as_millis(),
        files.len()
    );
}

#[test]
#[ignore]
fn benchmark_memory_profile() {
    println!("\n=== Memory Profile (Approximate) ===\n");
    println!("For accurate memory profiling, run with:");
    println!("  heaptrack cargo test -p codecs --features parquet --release -- benchmark_super_batch --ignored --nocapture");
    println!("\nOr use valgrind massif:");
    println!("  valgrind --tool=massif cargo test ...");
}

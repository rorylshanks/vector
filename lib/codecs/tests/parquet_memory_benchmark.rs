//! Memory and CPU benchmarks for Parquet encoding with JSON column expansion
//!
//! Run with: cargo test -p codecs --features parquet --release -- memory_benchmark --nocapture --ignored

use bytes::Bytes;
use codecs::encoding::format::{
    JsonColumnConfig, ParquetSerializer, ParquetSerializerConfig, SortingColumnConfig,
};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;
use vector_core::event::{Event, LogEvent, Value};

/// Get current memory usage from /proc/self/statm (Linux only)
/// Returns (virtual memory, resident set size) in bytes
fn get_memory_usage() -> (usize, usize) {
    if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
        let parts: Vec<&str> = statm.split_whitespace().collect();
        if parts.len() >= 2 {
            let page_size = 4096; // Typical page size
            let virt = parts[0].parse::<usize>().unwrap_or(0) * page_size;
            let rss = parts[1].parse::<usize>().unwrap_or(0) * page_size;
            return (virt, rss);
        }
    }
    (0, 0)
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Load events from real testdata log files
fn load_events_from_testdata(max_events: usize) -> Vec<Event> {
    let testdata_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("testdata");

    let mut events = Vec::new();

    if !testdata_dir.exists() {
        println!("Warning: testdata directory not found at {:?}", testdata_dir);
        return events;
    }

    // Read log files from testdata
    let mut log_files: Vec<_> = std::fs::read_dir(&testdata_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "log"))
        .collect();

    log_files.sort_by_key(|e| e.path());

    for entry in log_files {
        if events.len() >= max_events {
            break;
        }

        let file = match std::fs::File::open(entry.path()) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let reader = BufReader::new(file);
        for line in reader.lines() {
            if events.len() >= max_events {
                break;
            }

            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };

            if line.is_empty() {
                continue;
            }

            // Parse JSON and convert to LogEvent
            match serde_json::from_str::<serde_json::Value>(&line) {
                Ok(json) => {
                    let mut log = LogEvent::default();
                    if let serde_json::Value::Object(map) = json {
                        for (key, value) in map {
                            log.insert(key.as_str(), json_to_value(value));
                        }
                    }
                    events.push(Event::Log(log));
                }
                Err(_) => continue,
            }
        }
    }

    events
}

fn json_to_value(json: serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Boolean(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(ordered_float::NotNan::new(f).unwrap_or(ordered_float::NotNan::new(0.0).unwrap()))
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::Bytes(s.into()),
        serde_json::Value::Array(arr) => {
            Value::Array(arr.into_iter().map(json_to_value).collect())
        }
        serde_json::Value::Object(map) => {
            let mut obj = std::collections::BTreeMap::new();
            for (k, v) in map {
                obj.insert(k.into(), json_to_value(v));
            }
            Value::Object(obj)
        }
    }
}

fn create_test_events_with_varying_json(count: usize, overflow_keys_per_event: usize) -> Vec<Event> {
    (0..count)
        .map(|i| {
            let mut log = LogEvent::default();
            log.insert("id", i as i64);

            // Create JSON with configurable number of overflow keys
            let mut json_parts = vec![
                format!(r#""user_id":"user{}""#, i),
                format!(r#""session_id":"sess{}""#, i * 2),
                format!(r#""page_views":{}"#, i * 10),
                format!(r#""is_premium":{}"#, i % 2 == 0),
                format!(r#""score":{}"#, i as f64 * 1.5),
            ];

            // Add overflow keys
            for j in 0..overflow_keys_per_event {
                json_parts.push(format!(r#""extra_key_{}":"value_{}""#, j, i));
            }

            let json = format!("{{{}}}", json_parts.join(","));
            log.insert("properties", json);

            Event::Log(log)
        })
        .collect()
}

fn create_test_events_with_large_json(count: usize, json_size_bytes: usize) -> Vec<Event> {
    (0..count)
        .map(|i| {
            let mut log = LogEvent::default();
            log.insert("id", i as i64);

            // Create JSON with large string values
            let padding_size = json_size_bytes.saturating_sub(100);
            let padding: String = (0..padding_size).map(|_| 'x').collect();

            let json = format!(
                r#"{{"user_id":"user{}","large_field":"{}","count":{}}}"#,
                i, padding, i
            );
            log.insert("properties", json);

            Event::Log(log)
        })
        .collect()
}

#[test]
#[ignore]
fn memory_benchmark_varying_overflow_keys() {
    println!("\n=== Memory Benchmark: Varying Overflow Keys ===\n");
    println!("Testing how overflow key count affects memory usage\n");

    let event_count = 10000;
    let overflow_key_counts = [0, 10, 50, 100];

    println!("{:>15} {:>12} {:>12} {:>12} {:>12}",
        "Overflow Keys", "Time (ms)", "Peak RSS", "RSS Delta", "Output Size");
    println!("{}", "-".repeat(70));

    for &overflow_keys in &overflow_key_counts {
        // Force garbage collection by dropping previous data
        std::hint::black_box(());

        let (_, rss_before) = get_memory_usage();

        let events = create_test_events_with_varying_json(event_count, overflow_keys);
        let events_mem = std::mem::size_of_val(&events[..]);

        let mut config = ParquetSerializerConfig::default();
        config.infer_schema = true;
        config.rows_per_file = Some(2500);
        config.json_columns = Some(vec![JsonColumnConfig {
            column: "properties".to_string(),
            max_subcolumns: 10, // Only 10 subcolumns, rest go to overflow
            bucket_count: 8,
            max_depth: 5,
            keep_original_column: false,
            type_hints: None,
        }]);

        let serializer = ParquetSerializer::new(config).expect("Failed to create serializer");

        let start = Instant::now();
        let files = serializer.encode_batch_split(events).expect("Failed to encode");
        let elapsed = start.elapsed();

        let (_, rss_after) = get_memory_usage();
        let rss_delta = rss_after.saturating_sub(rss_before);

        let total_size: usize = files.iter().map(|f: &Bytes| f.len()).sum();

        println!("{:>15} {:>12.2} {:>12} {:>12} {:>12}",
            overflow_keys,
            elapsed.as_millis(),
            format_bytes(rss_after),
            format_bytes(rss_delta),
            format_bytes(total_size)
        );
    }
}

#[test]
#[ignore]
fn memory_benchmark_varying_event_count() {
    println!("\n=== Memory Benchmark: Varying Event Count ===\n");
    println!("Testing how event count affects memory usage with JSON expansion\n");

    let event_counts = [1000, 5000, 10000, 25000, 50000];

    println!("{:>12} {:>12} {:>12} {:>12} {:>15} {:>12}",
        "Events", "Time (ms)", "Peak RSS", "RSS Delta", "Throughput", "Output Size");
    println!("{}", "-".repeat(90));

    for &event_count in &event_counts {
        // Force cleanup
        std::hint::black_box(());

        let (_, rss_before) = get_memory_usage();

        let events = create_test_events_with_varying_json(event_count, 20);

        let mut config = ParquetSerializerConfig::default();
        config.infer_schema = true;
        config.rows_per_file = Some(5000);
        config.json_columns = Some(vec![JsonColumnConfig {
            column: "properties".to_string(),
            max_subcolumns: 50,
            bucket_count: 8,
            max_depth: 5,
            keep_original_column: true,
            type_hints: None,
        }]);

        let serializer = ParquetSerializer::new(config).expect("Failed to create serializer");

        let start = Instant::now();
        let files = serializer.encode_batch_split(events).expect("Failed to encode");
        let elapsed = start.elapsed();

        let (_, rss_after) = get_memory_usage();
        let rss_delta = rss_after.saturating_sub(rss_before);

        let total_size: usize = files.iter().map(|f: &Bytes| f.len()).sum();
        let throughput = event_count as f64 / elapsed.as_secs_f64();

        println!("{:>12} {:>12.2} {:>12} {:>12} {:>12.0}/s {:>12}",
            event_count,
            elapsed.as_millis(),
            format_bytes(rss_after),
            format_bytes(rss_delta),
            throughput,
            format_bytes(total_size)
        );
    }
}

#[test]
#[ignore]
fn memory_benchmark_large_json_values() {
    println!("\n=== Memory Benchmark: Large JSON Values ===\n");
    println!("Testing memory usage with varying JSON sizes per event\n");

    let event_count = 5000;
    let json_sizes = [100, 500, 1000, 5000, 10000];

    println!("{:>15} {:>12} {:>12} {:>12} {:>12}",
        "JSON Size", "Time (ms)", "Peak RSS", "RSS Delta", "Output Size");
    println!("{}", "-".repeat(70));

    for &json_size in &json_sizes {
        std::hint::black_box(());

        let (_, rss_before) = get_memory_usage();

        let events = create_test_events_with_large_json(event_count, json_size);

        let mut config = ParquetSerializerConfig::default();
        config.infer_schema = true;
        config.rows_per_file = Some(1000);
        config.json_columns = Some(vec![JsonColumnConfig {
            column: "properties".to_string(),
            max_subcolumns: 10,
            bucket_count: 4,
            max_depth: 3,
            keep_original_column: true,
            type_hints: None,
        }]);

        let serializer = ParquetSerializer::new(config).expect("Failed to create serializer");

        let start = Instant::now();
        let files = serializer.encode_batch_split(events).expect("Failed to encode");
        let elapsed = start.elapsed();

        let (_, rss_after) = get_memory_usage();
        let rss_delta = rss_after.saturating_sub(rss_before);

        let total_size: usize = files.iter().map(|f: &Bytes| f.len()).sum();

        println!("{:>12} B {:>12.2} {:>12} {:>12} {:>12}",
            json_size,
            elapsed.as_millis(),
            format_bytes(rss_after),
            format_bytes(rss_delta),
            format_bytes(total_size)
        );
    }
}

#[test]
#[ignore]
fn memory_benchmark_cpu_breakdown() {
    println!("\n=== CPU Time Breakdown ===\n");
    println!("Breaking down where CPU time is spent\n");

    let event_count = 10000;
    let iterations = 3;

    // Test 1: Event creation time
    let mut event_creation_times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let events = create_test_events_with_varying_json(event_count, 20);
        std::hint::black_box(&events);
        event_creation_times.push(start.elapsed());
    }
    let avg_event_creation = event_creation_times.iter().map(|d| d.as_millis()).sum::<u128>() / iterations as u128;

    // Test 2: Schema inference time
    let events = create_test_events_with_varying_json(event_count, 20);
    let mut schema_times = Vec::new();
    for _ in 0..iterations {
        let mut config = ParquetSerializerConfig::default();
        config.infer_schema = true;

        let start = Instant::now();
        let _serializer = ParquetSerializer::new(config).expect("Failed to create serializer");
        schema_times.push(start.elapsed());
    }
    let avg_schema = schema_times.iter().map(|d| d.as_micros()).sum::<u128>() / iterations as u128;

    // Test 3: Full encoding without JSON expansion
    let mut no_json_times = Vec::new();
    for _ in 0..iterations {
        let events = create_test_events_with_varying_json(event_count, 20);
        let mut config = ParquetSerializerConfig::default();
        config.infer_schema = true;
        config.rows_per_file = Some(2500);
        // No JSON column config

        let serializer = ParquetSerializer::new(config).expect("Failed to create serializer");

        let start = Instant::now();
        let _files = serializer.encode_batch_split(events).expect("Failed to encode");
        no_json_times.push(start.elapsed());
    }
    let avg_no_json = no_json_times.iter().map(|d| d.as_millis()).sum::<u128>() / iterations as u128;

    // Test 4: Full encoding with JSON expansion
    let mut with_json_times = Vec::new();
    for _ in 0..iterations {
        let events = create_test_events_with_varying_json(event_count, 20);
        let mut config = ParquetSerializerConfig::default();
        config.infer_schema = true;
        config.rows_per_file = Some(2500);
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
        let _files = serializer.encode_batch_split(events).expect("Failed to encode");
        with_json_times.push(start.elapsed());
    }
    let avg_with_json = with_json_times.iter().map(|d| d.as_millis()).sum::<u128>() / iterations as u128;

    // Test 5: With sorting
    let mut with_sorting_times = Vec::new();
    for _ in 0..iterations {
        let events = create_test_events_with_varying_json(event_count, 20);
        let mut config = ParquetSerializerConfig::default();
        config.infer_schema = true;
        config.rows_per_file = Some(2500);
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
        let _files = serializer.encode_batch_split(events).expect("Failed to encode");
        with_sorting_times.push(start.elapsed());
    }
    let avg_with_sorting = with_sorting_times.iter().map(|d| d.as_millis()).sum::<u128>() / iterations as u128;

    println!("Event count: {}", event_count);
    println!("Iterations: {}\n", iterations);

    println!("{:<40} {:>12}", "Operation", "Avg Time");
    println!("{}", "-".repeat(55));
    println!("{:<40} {:>12} ms", "Event creation", avg_event_creation);
    println!("{:<40} {:>12} µs", "Serializer creation (schema)", avg_schema);
    println!("{:<40} {:>12} ms", "Encoding (no JSON expansion)", avg_no_json);
    println!("{:<40} {:>12} ms", "Encoding (with JSON expansion)", avg_with_json);
    println!("{:<40} {:>12} ms", "Encoding (JSON + sorting)", avg_with_sorting);
    println!();
    println!("{:<40} {:>12} ms", "JSON expansion overhead", avg_with_json.saturating_sub(avg_no_json));
    println!("{:<40} {:>12} ms", "Sorting overhead", avg_with_sorting.saturating_sub(avg_with_json));
}

#[test]
#[ignore]
fn memory_benchmark_peak_memory_tracking() {
    println!("\n=== Peak Memory Tracking ===\n");
    println!("Tracking memory at each stage of processing\n");

    let event_count = 20000;

    let (_, rss_start) = get_memory_usage();
    println!("Initial RSS: {}", format_bytes(rss_start));

    // Stage 1: Create events
    let start = Instant::now();
    let events = create_test_events_with_varying_json(event_count, 30);
    let event_creation_time = start.elapsed();
    let (_, rss_after_events) = get_memory_usage();
    println!("After event creation: {} (+{}) - took {:?}",
        format_bytes(rss_after_events),
        format_bytes(rss_after_events.saturating_sub(rss_start)),
        event_creation_time
    );

    // Stage 2: Create serializer
    let mut config = ParquetSerializerConfig::default();
    config.infer_schema = true;
    config.rows_per_file = Some(5000);
    config.json_columns = Some(vec![JsonColumnConfig {
        column: "properties".to_string(),
        max_subcolumns: 50,
        bucket_count: 8,
        max_depth: 5,
        keep_original_column: true,
        type_hints: None,
    }]);

    let start = Instant::now();
    let serializer = ParquetSerializer::new(config).expect("Failed to create serializer");
    let serializer_creation_time = start.elapsed();
    let (_, rss_after_serializer) = get_memory_usage();
    println!("After serializer creation: {} (+{}) - took {:?}",
        format_bytes(rss_after_serializer),
        format_bytes(rss_after_serializer.saturating_sub(rss_after_events)),
        serializer_creation_time
    );

    // Stage 3: Encode
    let start = Instant::now();
    let files = serializer.encode_batch_split(events).expect("Failed to encode");
    let encoding_time = start.elapsed();
    let (_, rss_after_encoding) = get_memory_usage();
    println!("After encoding: {} (+{}) - took {:?}",
        format_bytes(rss_after_encoding),
        format_bytes(rss_after_encoding.saturating_sub(rss_after_serializer)),
        encoding_time
    );

    let total_output_size: usize = files.iter().map(|f: &Bytes| f.len()).sum();
    println!("\nOutput: {} files, {} total",
        files.len(),
        format_bytes(total_output_size)
    );

    // Drop files and check memory
    drop(files);
    let (_, rss_after_drop) = get_memory_usage();
    println!("After dropping output: {} (-{})",
        format_bytes(rss_after_drop),
        format_bytes(rss_after_encoding.saturating_sub(rss_after_drop))
    );

    println!("\nPeak memory delta: {}", format_bytes(rss_after_encoding.saturating_sub(rss_start)));
}

#[test]
#[ignore]
fn memory_benchmark_real_testdata() {
    println!("\n=== Memory Benchmark: Real Testdata ===\n");
    println!("Using actual log files from testdata directory\n");

    let event_counts = [1000, 5000, 10000];

    for &max_events in &event_counts {
        println!("--- Loading {} events from testdata ---", max_events);

        let (_, rss_start) = get_memory_usage();
        println!("Initial RSS: {}", format_bytes(rss_start));

        // Load events
        let start = Instant::now();
        let events = load_events_from_testdata(max_events);
        let load_time = start.elapsed();
        let actual_count = events.len();

        if actual_count == 0 {
            println!("No events loaded - skipping\n");
            continue;
        }

        let (_, rss_after_load) = get_memory_usage();
        println!("Loaded {} events in {:?}", actual_count, load_time);
        println!("Memory after loading: {} (+{})",
            format_bytes(rss_after_load),
            format_bytes(rss_after_load.saturating_sub(rss_start))
        );

        // Calculate average event size
        let sample_event = &events[0];
        if let Event::Log(log) = sample_event {
            if let Some(props) = log.get("properties") {
                let props_str = props.to_string();
                println!("Sample properties field size: {} bytes", props_str.len());
            }
        }

        // Configure serializer with JSON expansion
        let mut config = ParquetSerializerConfig::default();
        config.infer_schema = true;
        config.rows_per_file = Some(10000);
        config.json_columns = Some(vec![JsonColumnConfig {
            column: "properties".to_string(),
            max_subcolumns: 50,
            bucket_count: 8,
            max_depth: 5,
            keep_original_column: false,
            type_hints: None,
        }]);

        let serializer = ParquetSerializer::new(config).expect("Failed to create serializer");

        // Encode
        let start = Instant::now();
        let files = serializer.encode_batch_split(events).expect("Failed to encode");
        let encode_time = start.elapsed();

        let (_, rss_after_encode) = get_memory_usage();
        let total_output_size: usize = files.iter().map(|f: &Bytes| f.len()).sum();

        println!("Encoding took: {:?}", encode_time);
        println!("Memory after encoding: {} (+{})",
            format_bytes(rss_after_encode),
            format_bytes(rss_after_encode.saturating_sub(rss_after_load))
        );
        println!("Output: {} files, {} total", files.len(), format_bytes(total_output_size));
        println!("Throughput: {:.0} events/s", actual_count as f64 / encode_time.as_secs_f64());
        println!("Peak memory delta: {}\n", format_bytes(rss_after_encode.saturating_sub(rss_start)));

        drop(files);
    }
}

#[test]
#[ignore]
fn memory_benchmark_real_testdata_detailed() {
    println!("\n=== Detailed Memory Benchmark: Real Testdata ===\n");
    println!("Tracking memory at each stage with real data\n");

    let max_events = 10000;

    let (_, rss_start) = get_memory_usage();
    println!("Initial RSS: {}", format_bytes(rss_start));

    // Stage 1: Load events
    println!("\n--- Stage 1: Loading events ---");
    let start = Instant::now();
    let events = load_events_from_testdata(max_events);
    let load_time = start.elapsed();

    if events.is_empty() {
        println!("No events loaded from testdata - skipping test");
        return;
    }

    let (_, rss_after_load) = get_memory_usage();
    println!("Loaded {} events in {:?}", events.len(), load_time);
    println!("Memory: {} (+{})",
        format_bytes(rss_after_load),
        format_bytes(rss_after_load.saturating_sub(rss_start))
    );
    println!("Memory per event: {} bytes", (rss_after_load - rss_start) / events.len());

    // Stage 2: Create serializer
    println!("\n--- Stage 2: Creating serializer ---");
    let mut config = ParquetSerializerConfig::default();
    config.infer_schema = true;
    config.rows_per_file = Some(10000);
    config.json_columns = Some(vec![JsonColumnConfig {
        column: "properties".to_string(),
        max_subcolumns: 50,
        bucket_count: 8,
        max_depth: 5,
        keep_original_column: false,
        type_hints: None,
    }]);

    let start = Instant::now();
    let serializer = ParquetSerializer::new(config).expect("Failed to create serializer");
    println!("Serializer created in {:?}", start.elapsed());

    let (_, rss_after_serializer) = get_memory_usage();
    println!("Memory: {} (+{})",
        format_bytes(rss_after_serializer),
        format_bytes(rss_after_serializer.saturating_sub(rss_after_load))
    );

    // Stage 3: Encode
    println!("\n--- Stage 3: Encoding ---");
    let start = Instant::now();
    let files = serializer.encode_batch_split(events).expect("Failed to encode");
    let encode_time = start.elapsed();

    let (_, rss_after_encode) = get_memory_usage();
    let total_output_size: usize = files.iter().map(|f: &Bytes| f.len()).sum();

    println!("Encoding took: {:?}", encode_time);
    println!("Memory: {} (+{})",
        format_bytes(rss_after_encode),
        format_bytes(rss_after_encode.saturating_sub(rss_after_serializer))
    );
    println!("Output: {} files, {} total", files.len(), format_bytes(total_output_size));

    // Stage 4: Cleanup
    println!("\n--- Stage 4: Cleanup ---");
    drop(files);
    let (_, rss_after_drop) = get_memory_usage();
    println!("Memory after dropping output: {} (-{})",
        format_bytes(rss_after_drop),
        format_bytes(rss_after_encode.saturating_sub(rss_after_drop))
    );

    println!("\n=== Summary ===");
    println!("Peak memory: {}", format_bytes(rss_after_encode));
    println!("Peak delta from start: {}", format_bytes(rss_after_encode.saturating_sub(rss_start)));
}

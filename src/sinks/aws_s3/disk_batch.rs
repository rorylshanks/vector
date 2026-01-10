//! Disk-backed batching for high-cardinality S3 partitioning.
//!
//! This module provides disk-backed batching for scenarios with high partition cardinality
//! (e.g., tens of thousands of unique `key_prefix` values) and large batch sizes (7-10GB).
//!
//! Instead of keeping all partition batches in memory, events are serialized to temporary
//! files on disk. When a batch is ready (by size, count, or timeout), the file is read
//! back and processed.

use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    hash::Hash,
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    pin::Pin,
    sync::atomic::{AtomicU64, Ordering},
    task::{Context, Poll},
    time::Duration,
};

use bytes::{Bytes, BytesMut};
use futures::stream::{Fuse, Stream, StreamExt};
use pin_project::pin_project;
use tokio_util::time::{delay_queue::Key, DelayQueue};
use vector_lib::{
    ByteSizeOf,
    event::{Event, EventContainer},
    finalization::{EventFinalizers, Finalizable},
    partition::Partitioner,
};

use crate::event::EventArray;

/// Counter for generating unique batch file names.
static BATCH_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Configuration for disk-backed batching.
#[derive(Debug, Clone)]
pub struct DiskBatchConfig {
    /// Directory to store temporary batch files.
    pub temp_dir: PathBuf,
    /// Maximum number of bytes per batch before flushing.
    pub max_bytes: usize,
    /// Maximum number of events per batch before flushing.
    pub max_events: usize,
    /// Timeout after which a batch is flushed regardless of size.
    pub timeout: Duration,
}

impl Default for DiskBatchConfig {
    fn default() -> Self {
        Self {
            temp_dir: std::env::temp_dir().join("vector-s3-batches"),
            max_bytes: 10 * 1024 * 1024 * 1024, // 10GB
            max_events: 100_000,
            timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// State for a single partition's batch being accumulated on disk.
#[derive(Debug)]
pub struct DiskPartitionState {
    /// Path to the temporary file for this partition.
    file_path: PathBuf,
    /// Buffered writer for the file.
    writer: Option<BufWriter<File>>,
    /// Number of events written to this partition.
    event_count: usize,
    /// Total bytes written (serialized size).
    bytes_written: u64,
    /// Byte size for memory accounting (original event sizes).
    events_byte_size: usize,
    /// Event finalizers accumulated for this batch.
    finalizers: EventFinalizers,
}

impl DiskPartitionState {
    /// Creates a new partition state with a new temp file.
    pub fn new(temp_dir: &Path) -> io::Result<Self> {
        let file_id = BATCH_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let file_name = format!("batch-{}-{}.tmp", std::process::id(), file_id);
        let file_path = temp_dir.join(file_name);

        tracing::debug!(
            file_path = %file_path.display(),
            "Creating new disk partition state"
        );

        // Create parent directory if needed
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&file_path)?;

        Ok(Self {
            file_path,
            writer: Some(BufWriter::with_capacity(256 * 1024, file)), // 256KB buffer
            event_count: 0,
            bytes_written: 0,
            events_byte_size: 0,
            finalizers: EventFinalizers::default(),
        })
    }

    /// Writes an event to the partition's temp file.
    ///
    /// Events are stored as length-prefixed protobuf-encoded EventArrays.
    /// Format: [u32 length][protobuf bytes]...
    pub fn write_event(&mut self, event: Event) -> io::Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Writer already closed"))?;

        // Track byte size before we move the event
        let event_byte_size = event.size_of();

        // Take finalizers from the event
        let mut events = vec![event];
        self.finalizers.merge(events.take_finalizers());
        let event = events.pop().unwrap();

        // Convert to EventArray and encode as protobuf
        let event_array = EventArray::from(event);
        let proto: crate::event::proto::EventArray = event_array.into();

        // Encode to bytes using prost
        let mut buf = BytesMut::new();
        prost::Message::encode(&proto, &mut buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Write length prefix (u32 big-endian) followed by data
        let len = buf.len() as u32;
        writer.write_all(&len.to_be_bytes())?;
        writer.write_all(&buf)?;

        self.event_count += 1;
        self.bytes_written += 4 + buf.len() as u64;
        self.events_byte_size += event_byte_size;

        Ok(())
    }

    /// Flushes the writer to disk.
    #[allow(dead_code)]
    pub fn flush(&mut self) -> io::Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            writer.flush()?;
        }
        Ok(())
    }

    /// Returns the number of events in this partition.
    pub fn event_count(&self) -> usize {
        self.event_count
    }

    /// Returns the total bytes written to disk.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Returns the accumulated byte size of events (for memory accounting).
    #[allow(dead_code)]
    pub fn events_byte_size(&self) -> usize {
        self.events_byte_size
    }

    /// Finalizes the partition and returns a DiskBatch for processing.
    ///
    /// This closes the file writer and prepares the batch for reading.
    pub fn take_batch(&mut self) -> io::Result<DiskBatch> {
        tracing::debug!(
            file_path = %self.file_path.display(),
            event_count = self.event_count,
            bytes_written = self.bytes_written,
            "Taking batch from partition"
        );

        // Flush and close the writer
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
        }

        let batch = DiskBatch {
            file_path: self.file_path.clone(),
            event_count: self.event_count,
            bytes_on_disk: self.bytes_written,
            events_byte_size: self.events_byte_size,
            finalizers: std::mem::take(&mut self.finalizers),
        };

        // Reset state for reuse
        self.event_count = 0;
        self.bytes_written = 0;
        self.events_byte_size = 0;

        Ok(batch)
    }

    /// Cleans up the temp file without returning a batch.
    #[allow(dead_code)]
    pub fn cleanup(mut self) {
        // Drop the writer first
        drop(self.writer.take());
        // Try to remove the file
        let _ = fs::remove_file(&self.file_path);
    }
}

impl Drop for DiskPartitionState {
    fn drop(&mut self) {
        // Clean up temp file on drop
        drop(self.writer.take());
        let _ = fs::remove_file(&self.file_path);
    }
}

/// A completed batch stored on disk, ready for processing.
#[derive(Debug)]
pub struct DiskBatch {
    /// Path to the temporary file containing the batch.
    pub file_path: PathBuf,
    /// Number of events in the batch.
    pub event_count: usize,
    /// Total bytes on disk.
    pub bytes_on_disk: u64,
    /// Byte size for telemetry/accounting.
    pub events_byte_size: usize,
    /// Finalizers for acknowledgment.
    pub finalizers: EventFinalizers,
}

impl DiskBatch {
    /// Reads all events from the batch file.
    ///
    /// This deserializes the events from the temp file and returns them as a Vec.
    pub fn read_events(&self) -> io::Result<Vec<Event>> {
        let file = File::open(&self.file_path)?;
        let mut reader = BufReader::with_capacity(256 * 1024, file);
        let mut events = Vec::with_capacity(self.event_count);

        let mut len_buf = [0u8; 4];
        loop {
            // Read length prefix
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let len = u32::from_be_bytes(len_buf) as usize;

            // Read the protobuf data
            let mut data = vec![0u8; len];
            reader.read_exact(&mut data)?;

            // Decode the protobuf
            let proto: crate::event::proto::EventArray =
                prost::Message::decode(Bytes::from(data))
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            // Convert to EventArray and extract events
            let event_array: EventArray = proto.into();
            for event in event_array.into_events() {
                events.push(event);
            }
        }

        Ok(events)
    }

    /// Cleans up the batch file after processing.
    pub fn cleanup(self) {
        let _ = fs::remove_file(&self.file_path);
    }
}

impl Finalizable for DiskBatch {
    fn take_finalizers(&mut self) -> EventFinalizers {
        std::mem::take(&mut self.finalizers)
    }
}

/// A disk-backed partitioned batcher that stores events on disk per partition.
///
/// This is designed for high-cardinality partitioning with large batch sizes.
/// Instead of keeping all events in memory, they are written to temp files.
#[pin_project]
pub struct DiskBackedPartitionedBatcher<St, Prt>
where
    Prt: Partitioner,
{
    /// Configuration for disk batching.
    config: DiskBatchConfig,
    /// The store of live partition states.
    partitions: HashMap<Prt::Key, DiskPartitionState>,
    /// Closed batches ready for output.
    closed_batches: Vec<(Prt::Key, DiskBatch)>,
    /// Expiration queue for batch timeouts.
    expirations: DelayQueue<Prt::Key>,
    /// Mapping from partition key to expiration key.
    expiration_map: HashMap<Prt::Key, Key>,
    /// The partitioner.
    partitioner: Prt,
    /// The input stream.
    #[pin]
    stream: Fuse<St>,
}

impl<St, Prt> DiskBackedPartitionedBatcher<St, Prt>
where
    St: Stream<Item = Prt::Item>,
    Prt: Partitioner + Unpin,
    Prt::Key: Eq + Hash + Clone,
    Prt::Item: ByteSizeOf,
{
    /// Creates a new disk-backed partitioned batcher.
    pub fn new(stream: St, partitioner: Prt, config: DiskBatchConfig) -> Self {
        Self {
            config,
            partitions: HashMap::new(),
            closed_batches: Vec::new(),
            expirations: DelayQueue::new(),
            expiration_map: HashMap::new(),
            partitioner,
            stream: stream.fuse(),
        }
    }
}

impl<St, Prt> Stream for DiskBackedPartitionedBatcher<St, Prt>
where
    St: Stream<Item = Event>,
    Prt: Partitioner<Item = Event> + Unpin,
    Prt::Key: Eq + Hash + Clone,
{
    type Item = (Prt::Key, DiskBatch);

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            // First, return any closed batches
            if let Some((_key, batch)) = this.closed_batches.last() {
                tracing::debug!(
                    event_count = batch.event_count,
                    bytes_on_disk = batch.bytes_on_disk,
                    "Returning closed batch"
                );
                let batch = this.closed_batches.pop().unwrap();
                return Poll::Ready(Some(batch));
            }

            // Poll the input stream
            match this.stream.as_mut().poll_next(cx) {
                Poll::Pending => {
                    tracing::trace!(
                        partition_count = this.partitions.len(),
                        expiration_count = this.expirations.len(),
                        "Input stream pending, checking expirations"
                    );
                    // Check for expired batches while pending
                    match this.expirations.poll_expired(cx) {
                        Poll::Ready(Some(expired)) => {
                            let key = expired.into_inner();
                            tracing::debug!("Batch timeout expired, flushing partition");
                            this.expiration_map.remove(&key);

                            if let Some(mut state) = this.partitions.remove(&key) {
                                match state.take_batch() {
                                    Ok(batch) => {
                                        tracing::debug!(
                                            event_count = batch.event_count,
                                            "Batch ready from timeout"
                                        );
                                        this.closed_batches.push((key, batch));
                                    }
                                    Err(e) => {
                                        tracing::error!(error = %e, "Failed to take batch from partition");
                                    }
                                }
                            }
                            continue;
                        }
                        Poll::Ready(None) => {
                            tracing::trace!("No expirations scheduled, returning pending");
                            return Poll::Pending;
                        }
                        Poll::Pending => {
                            tracing::trace!("Expirations pending, returning pending");
                            return Poll::Pending;
                        }
                    }
                }
                Poll::Ready(None) => {
                    tracing::debug!(
                        partition_count = this.partitions.len(),
                        "Input stream ended, flushing remaining partitions"
                    );
                    // Stream ended, flush all remaining partitions
                    if !this.partitions.is_empty() {
                        this.expirations.clear();
                        this.expiration_map.clear();

                        let keys: Vec<_> = this.partitions.keys().cloned().collect();
                        for key in keys {
                            if let Some(mut state) = this.partitions.remove(&key) {
                                if state.event_count() > 0 {
                                    tracing::debug!(
                                        event_count = state.event_count(),
                                        "Flushing partition on stream end"
                                    );
                                    match state.take_batch() {
                                        Ok(batch) => {
                                            this.closed_batches.push((key, batch));
                                        }
                                        Err(e) => {
                                            tracing::error!(error = %e, "Failed to take batch from partition");
                                        }
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    tracing::debug!("All partitions flushed, stream complete");
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(event)) => {
                    // Partition the event
                    let key = this.partitioner.partition(&event);

                    // Get or create partition state
                    let state = if let Some(state) = this.partitions.get_mut(&key) {
                        state
                    } else {
                        tracing::debug!(
                            partition_count = this.partitions.len() + 1,
                            timeout_secs = this.config.timeout.as_secs(),
                            "Creating new partition"
                        );
                        // Create new partition state
                        match DiskPartitionState::new(&this.config.temp_dir) {
                            Ok(state) => {
                                this.partitions.insert(key.clone(), state);

                                // Set up expiration timer
                                let exp_key =
                                    this.expirations.insert(key.clone(), this.config.timeout);
                                this.expiration_map.insert(key.clone(), exp_key);

                                this.partitions.get_mut(&key).unwrap()
                            }
                            Err(e) => {
                                tracing::error!(error = %e, "Failed to create partition state");
                                continue;
                            }
                        }
                    };

                    // Write the event
                    if let Err(e) = state.write_event(event) {
                        tracing::error!(error = %e, "Failed to write event to disk batch");
                        continue;
                    }

                    // Check if batch is ready
                    let event_count = state.event_count();
                    let bytes_written = state.bytes_written() as usize;
                    let should_flush =
                        event_count >= this.config.max_events || bytes_written >= this.config.max_bytes;

                    // Log progress periodically
                    if event_count % 10000 == 0 {
                        tracing::debug!(
                            event_count,
                            bytes_written,
                            max_events = this.config.max_events,
                            max_bytes = this.config.max_bytes,
                            "Partition progress"
                        );
                    }

                    if should_flush {
                        tracing::info!(
                            event_count,
                            bytes_written,
                            "Batch size/count limit reached, flushing"
                        );
                        // Remove from expiration tracking
                        if let Some(exp_key) = this.expiration_map.remove(&key) {
                            this.expirations.remove(&exp_key);
                        }

                        if let Some(mut state) = this.partitions.remove(&key) {
                            match state.take_batch() {
                                Ok(batch) => {
                                    this.closed_batches.push((key, batch));
                                }
                                Err(e) => {
                                    tracing::error!(error = %e, "Failed to take batch from partition");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.stream.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vector_lib::event::{LogEvent, Value};

    fn create_test_event(id: u64) -> Event {
        let mut log = LogEvent::default();
        log.insert("id", Value::from(id));
        log.insert("message", Value::from(format!("Test message {}", id)));
        Event::Log(log)
    }

    #[test]
    fn test_disk_partition_state_write_and_read() {
        let temp_dir = std::env::temp_dir().join("vector-test-batches");
        let _ = fs::create_dir_all(&temp_dir);

        let mut state = DiskPartitionState::new(&temp_dir).unwrap();

        // Write some events
        for i in 0..10 {
            state.write_event(create_test_event(i)).unwrap();
        }

        assert_eq!(state.event_count(), 10);

        // Take the batch
        let batch = state.take_batch().unwrap();
        assert_eq!(batch.event_count, 10);

        // Read events back
        let events = batch.read_events().unwrap();
        assert_eq!(events.len(), 10);

        // Verify content
        for (i, event) in events.iter().enumerate() {
            if let Event::Log(log) = event {
                assert_eq!(log.get("id").unwrap(), &Value::from(i as u64));
            } else {
                panic!("Expected log event");
            }
        }

        // Cleanup
        batch.cleanup();
    }

    #[test]
    fn test_disk_partition_state_cleanup_on_drop() {
        let temp_dir = std::env::temp_dir().join("vector-test-batches-drop");
        let _ = fs::create_dir_all(&temp_dir);

        let file_path;
        {
            let mut state = DiskPartitionState::new(&temp_dir).unwrap();
            file_path = state.file_path.clone();
            state.write_event(create_test_event(1)).unwrap();
            // State dropped here
        }

        // File should be cleaned up
        assert!(!file_path.exists());
    }
}

//! Super-batch sink for S3 that supports parquet with rows_per_file splitting.
//!
//! This sink uses IncrementalRequestBuilder to split a batch of events into
//! multiple parquet files based on the rows_per_file configuration.

use std::{fmt, hash::Hash, num::NonZeroUsize};

use futures::{StreamExt, stream::BoxStream};
use tower::Service;
use vector_lib::{ByteSizeOf, event::Event, partition::Partitioner, stream::DriverResponse};

use crate::sinks::{
    prelude::*,
    s3_common::{partitioner::S3PartitionKey, service::S3Request},
    util::SinkBuilderExt,
};

use super::sink::S3SuperBatchRequestBuilder;

pub struct S3SuperBatchSink<Svc, P> {
    service: Svc,
    request_builder: S3SuperBatchRequestBuilder,
    partitioner: P,
    batcher_settings: BatcherSettings,
    concurrency_limit: NonZeroUsize,
}

impl<Svc, P> S3SuperBatchSink<Svc, P> {
    pub fn new(
        service: Svc,
        request_builder: S3SuperBatchRequestBuilder,
        partitioner: P,
        batcher_settings: BatcherSettings,
        concurrency_limit: NonZeroUsize,
    ) -> Self {
        Self {
            partitioner,
            service,
            request_builder,
            batcher_settings,
            concurrency_limit,
        }
    }
}

impl<Svc, P> S3SuperBatchSink<Svc, P>
where
    Svc: Service<S3Request> + Send + 'static,
    Svc::Future: Send + 'static,
    Svc::Response: DriverResponse + Send + 'static,
    Svc::Error: fmt::Debug + Into<crate::Error> + Send,
    P: Partitioner<Item = Event, Key = Option<S3PartitionKey>> + Unpin + Send,
    P::Key: Eq + Hash + Clone,
    P::Item: ByteSizeOf,
{
    async fn run_inner(self: Box<Self>, input: BoxStream<'_, Event>) -> Result<(), ()> {
        let partitioner = self.partitioner;
        let settings = self.batcher_settings;
        let request_builder = self.request_builder;
        let concurrency_limit = self.concurrency_limit;

        input
            .batched_partitioned(partitioner, || settings.as_byte_size_config())
            .filter_map(|(key, batch)| async move { key.map(move |k| (k, batch)) })
            .concurrent_incremental_request_builder(concurrency_limit, request_builder)
            .flat_map(futures::stream::iter)
            .filter_map(|request| async move {
                match request {
                    Err(error) => {
                        emit!(SinkRequestBuildError { error });
                        None
                    }
                    Ok(req) => Some(req),
                }
            })
            .into_driver(self.service)
            .run()
            .await
    }
}

#[async_trait::async_trait]
impl<Svc, P> StreamSink<Event> for S3SuperBatchSink<Svc, P>
where
    Svc: Service<S3Request> + Send + 'static,
    Svc::Future: Send + 'static,
    Svc::Response: DriverResponse + Send + 'static,
    Svc::Error: fmt::Debug + Into<crate::Error> + Send,
    P: Partitioner<Item = Event, Key = Option<S3PartitionKey>> + Unpin + Send,
    P::Key: Eq + Hash + Clone,
    P::Item: ByteSizeOf,
{
    async fn run(mut self: Box<Self>, input: BoxStream<'_, Event>) -> Result<(), ()> {
        self.run_inner(input).await
    }
}

use conjure_http::endpoint;
#[conjure_http::conjure_client(name = "HandoverService")]
pub trait HandoverService<
    #[response_body]
    I: Iterator<
            Item = Result<conjure_http::private::Bytes, conjure_http::private::Error>,
        >,
> {
    #[endpoint(
        method = POST,
        path = "/handover/sync",
        name = "syncContext",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    fn sync_context(
        &self,
        #[body(serializer = conjure_http::client::StdRequestSerializer)]
        request: &super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverRequest,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverResponse,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = POST,
        path = "/handover/stream",
        name = "streamHandover",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    fn stream_handover(
        &self,
        #[body(serializer = conjure_http::client::StdRequestSerializer)]
        handover: &super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverRequest,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverStatus,
        conjure_http::private::Error,
    >;
}
#[conjure_http::conjure_client(name = "HandoverService")]
pub trait AsyncHandoverService<
    #[response_body]
    I: conjure_http::private::Stream<
            Item = Result<conjure_http::private::Bytes, conjure_http::private::Error>,
        >,
> {
    #[endpoint(
        method = POST,
        path = "/handover/sync",
        name = "syncContext",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    async fn sync_context(
        &self,
        #[body(serializer = conjure_http::client::StdRequestSerializer)]
        request: &super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverRequest,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverResponse,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = POST,
        path = "/handover/stream",
        name = "streamHandover",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    async fn stream_handover(
        &self,
        #[body(serializer = conjure_http::client::StdRequestSerializer)]
        handover: &super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverRequest,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverStatus,
        conjure_http::private::Error,
    >;
}
#[conjure_http::conjure_client(name = "HandoverService", local)]
pub trait LocalAsyncHandoverService<
    #[response_body]
    I: conjure_http::private::Stream<
            Item = Result<conjure_http::private::Bytes, conjure_http::private::Error>,
        >,
> {
    #[endpoint(
        method = POST,
        path = "/handover/sync",
        name = "syncContext",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    async fn sync_context(
        &self,
        #[body(serializer = conjure_http::client::StdRequestSerializer)]
        request: &super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverRequest,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverResponse,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = POST,
        path = "/handover/stream",
        name = "streamHandover",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    async fn stream_handover(
        &self,
        #[body(serializer = conjure_http::client::StdRequestSerializer)]
        handover: &super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverRequest,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverStatus,
        conjure_http::private::Error,
    >;
}

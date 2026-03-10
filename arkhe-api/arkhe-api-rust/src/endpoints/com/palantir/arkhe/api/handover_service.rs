use conjure_http::endpoint;
#[conjure_http::conjure_endpoints(
    name = "HandoverService",
    use_legacy_error_serialization
)]
pub trait HandoverService {
    #[endpoint(
        method = POST,
        path = "/handover/sync",
        name = "syncContext",
        produces = conjure_http::server::StdResponseSerializer
    )]
    fn sync_context(
        &self,
        #[body(deserializer = conjure_http::server::StdRequestDeserializer)]
        request: super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverRequest,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverResponse,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = POST,
        path = "/handover/stream",
        name = "streamHandover",
        produces = conjure_http::server::StdResponseSerializer
    )]
    fn stream_handover(
        &self,
        #[body(deserializer = conjure_http::server::StdRequestDeserializer)]
        handover: super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverRequest,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverStatus,
        conjure_http::private::Error,
    >;
}
#[conjure_http::conjure_endpoints(
    name = "HandoverService",
    use_legacy_error_serialization
)]
pub trait AsyncHandoverService {
    #[endpoint(
        method = POST,
        path = "/handover/sync",
        name = "syncContext",
        produces = conjure_http::server::StdResponseSerializer
    )]
    async fn sync_context(
        &self,
        #[body(deserializer = conjure_http::server::StdRequestDeserializer)]
        request: super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverRequest,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverResponse,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = POST,
        path = "/handover/stream",
        name = "streamHandover",
        produces = conjure_http::server::StdResponseSerializer
    )]
    async fn stream_handover(
        &self,
        #[body(deserializer = conjure_http::server::StdRequestDeserializer)]
        handover: super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverRequest,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverStatus,
        conjure_http::private::Error,
    >;
}
#[conjure_http::conjure_endpoints(
    name = "HandoverService",
    use_legacy_error_serialization,
    local
)]
pub trait LocalAsyncHandoverService {
    #[endpoint(
        method = POST,
        path = "/handover/sync",
        name = "syncContext",
        produces = conjure_http::server::StdResponseSerializer
    )]
    async fn sync_context(
        &self,
        #[body(deserializer = conjure_http::server::StdRequestDeserializer)]
        request: super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverRequest,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverResponse,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = POST,
        path = "/handover/stream",
        name = "streamHandover",
        produces = conjure_http::server::StdResponseSerializer
    )]
    async fn stream_handover(
        &self,
        #[body(deserializer = conjure_http::server::StdRequestDeserializer)]
        handover: super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverRequest,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::HandoverStatus,
        conjure_http::private::Error,
    >;
}

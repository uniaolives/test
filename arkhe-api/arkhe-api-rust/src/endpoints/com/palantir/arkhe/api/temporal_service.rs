use conjure_http::endpoint;
#[conjure_http::conjure_endpoints(
    name = "TemporalService",
    use_legacy_error_serialization
)]
pub trait TemporalService {
    #[endpoint(
        method = POST,
        path = "/temporal/emit",
        name = "emit",
        produces = conjure_http::server::StdResponseSerializer
    )]
    fn emit(
        &self,
        #[body(deserializer = conjure_http::server::StdRequestDeserializer)]
        orb: super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::CoherenceMetric,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = GET,
        path = "/temporal/observe/{id}",
        name = "observe",
        produces = conjure_http::server::StdResponseSerializer
    )]
    fn observe(
        &self,
        #[path(name = "id", decoder = conjure_http::server::conjure::FromPlainDecoder)]
        id: String,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = POST,
        path = "/temporal/collapse/{id}",
        name = "collapse",
        produces = conjure_http::server::StdResponseSerializer
    )]
    fn collapse(
        &self,
        #[path(name = "id", decoder = conjure_http::server::conjure::FromPlainDecoder)]
        id: String,
    ) -> Result<bool, conjure_http::private::Error>;
    #[endpoint(
        method = POST,
        path = "/temporal/quantize",
        name = "quantize",
        produces = conjure_http::server::conjure::CollectionResponseSerializer
    )]
    fn quantize(
        &self,
        #[body(deserializer = conjure_http::server::StdRequestDeserializer)]
        trajectory: super::super::super::super::super::super::objects::com::palantir::arkhe::api::TemporalTrajectory,
    ) -> Result<
        Vec<
            super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
        >,
        conjure_http::private::Error,
    >;
}
#[conjure_http::conjure_endpoints(
    name = "TemporalService",
    use_legacy_error_serialization
)]
pub trait AsyncTemporalService {
    #[endpoint(
        method = POST,
        path = "/temporal/emit",
        name = "emit",
        produces = conjure_http::server::StdResponseSerializer
    )]
    async fn emit(
        &self,
        #[body(deserializer = conjure_http::server::StdRequestDeserializer)]
        orb: super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::CoherenceMetric,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = GET,
        path = "/temporal/observe/{id}",
        name = "observe",
        produces = conjure_http::server::StdResponseSerializer
    )]
    async fn observe(
        &self,
        #[path(name = "id", decoder = conjure_http::server::conjure::FromPlainDecoder)]
        id: String,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = POST,
        path = "/temporal/collapse/{id}",
        name = "collapse",
        produces = conjure_http::server::StdResponseSerializer
    )]
    async fn collapse(
        &self,
        #[path(name = "id", decoder = conjure_http::server::conjure::FromPlainDecoder)]
        id: String,
    ) -> Result<bool, conjure_http::private::Error>;
    #[endpoint(
        method = POST,
        path = "/temporal/quantize",
        name = "quantize",
        produces = conjure_http::server::conjure::CollectionResponseSerializer
    )]
    async fn quantize(
        &self,
        #[body(deserializer = conjure_http::server::StdRequestDeserializer)]
        trajectory: super::super::super::super::super::super::objects::com::palantir::arkhe::api::TemporalTrajectory,
    ) -> Result<
        Vec<
            super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
        >,
        conjure_http::private::Error,
    >;
}
#[conjure_http::conjure_endpoints(
    name = "TemporalService",
    use_legacy_error_serialization,
    local
)]
pub trait LocalAsyncTemporalService {
    #[endpoint(
        method = POST,
        path = "/temporal/emit",
        name = "emit",
        produces = conjure_http::server::StdResponseSerializer
    )]
    async fn emit(
        &self,
        #[body(deserializer = conjure_http::server::StdRequestDeserializer)]
        orb: super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::CoherenceMetric,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = GET,
        path = "/temporal/observe/{id}",
        name = "observe",
        produces = conjure_http::server::StdResponseSerializer
    )]
    async fn observe(
        &self,
        #[path(name = "id", decoder = conjure_http::server::conjure::FromPlainDecoder)]
        id: String,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = POST,
        path = "/temporal/collapse/{id}",
        name = "collapse",
        produces = conjure_http::server::StdResponseSerializer
    )]
    async fn collapse(
        &self,
        #[path(name = "id", decoder = conjure_http::server::conjure::FromPlainDecoder)]
        id: String,
    ) -> Result<bool, conjure_http::private::Error>;
    #[endpoint(
        method = POST,
        path = "/temporal/quantize",
        name = "quantize",
        produces = conjure_http::server::conjure::CollectionResponseSerializer
    )]
    async fn quantize(
        &self,
        #[body(deserializer = conjure_http::server::StdRequestDeserializer)]
        trajectory: super::super::super::super::super::super::objects::com::palantir::arkhe::api::TemporalTrajectory,
    ) -> Result<
        Vec<
            super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
        >,
        conjure_http::private::Error,
    >;
}

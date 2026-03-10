use conjure_http::endpoint;
#[conjure_http::conjure_client(name = "TemporalService")]
pub trait TemporalService<
    #[response_body]
    I: Iterator<
            Item = Result<conjure_http::private::Bytes, conjure_http::private::Error>,
        >,
> {
    #[endpoint(
        method = POST,
        path = "/temporal/emit",
        name = "emit",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    fn emit(
        &self,
        #[body(serializer = conjure_http::client::StdRequestSerializer)]
        orb: &super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::CoherenceMetric,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = GET,
        path = "/temporal/observe/{id}",
        name = "observe",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    fn observe(
        &self,
        #[path(name = "id", encoder = conjure_http::client::conjure::PlainEncoder)]
        id: &str,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = POST,
        path = "/temporal/collapse/{id}",
        name = "collapse",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    fn collapse(
        &self,
        #[path(name = "id", encoder = conjure_http::client::conjure::PlainEncoder)]
        id: &str,
    ) -> Result<bool, conjure_http::private::Error>;
    #[endpoint(
        method = POST,
        path = "/temporal/quantize",
        name = "quantize",
        accept = conjure_http::client::conjure::CollectionResponseDeserializer
    )]
    fn quantize(
        &self,
        #[body(serializer = conjure_http::client::StdRequestSerializer)]
        trajectory: &super::super::super::super::super::super::objects::com::palantir::arkhe::api::TemporalTrajectory,
    ) -> Result<
        Vec<
            super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
        >,
        conjure_http::private::Error,
    >;
}
#[conjure_http::conjure_client(name = "TemporalService")]
pub trait AsyncTemporalService<
    #[response_body]
    I: conjure_http::private::Stream<
            Item = Result<conjure_http::private::Bytes, conjure_http::private::Error>,
        >,
> {
    #[endpoint(
        method = POST,
        path = "/temporal/emit",
        name = "emit",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    async fn emit(
        &self,
        #[body(serializer = conjure_http::client::StdRequestSerializer)]
        orb: &super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::CoherenceMetric,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = GET,
        path = "/temporal/observe/{id}",
        name = "observe",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    async fn observe(
        &self,
        #[path(name = "id", encoder = conjure_http::client::conjure::PlainEncoder)]
        id: &str,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = POST,
        path = "/temporal/collapse/{id}",
        name = "collapse",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    async fn collapse(
        &self,
        #[path(name = "id", encoder = conjure_http::client::conjure::PlainEncoder)]
        id: &str,
    ) -> Result<bool, conjure_http::private::Error>;
    #[endpoint(
        method = POST,
        path = "/temporal/quantize",
        name = "quantize",
        accept = conjure_http::client::conjure::CollectionResponseDeserializer
    )]
    async fn quantize(
        &self,
        #[body(serializer = conjure_http::client::StdRequestSerializer)]
        trajectory: &super::super::super::super::super::super::objects::com::palantir::arkhe::api::TemporalTrajectory,
    ) -> Result<
        Vec<
            super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
        >,
        conjure_http::private::Error,
    >;
}
#[conjure_http::conjure_client(name = "TemporalService", local)]
pub trait LocalAsyncTemporalService<
    #[response_body]
    I: conjure_http::private::Stream<
            Item = Result<conjure_http::private::Bytes, conjure_http::private::Error>,
        >,
> {
    #[endpoint(
        method = POST,
        path = "/temporal/emit",
        name = "emit",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    async fn emit(
        &self,
        #[body(serializer = conjure_http::client::StdRequestSerializer)]
        orb: &super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::CoherenceMetric,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = GET,
        path = "/temporal/observe/{id}",
        name = "observe",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    async fn observe(
        &self,
        #[path(name = "id", encoder = conjure_http::client::conjure::PlainEncoder)]
        id: &str,
    ) -> Result<
        super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
        conjure_http::private::Error,
    >;
    #[endpoint(
        method = POST,
        path = "/temporal/collapse/{id}",
        name = "collapse",
        accept = conjure_http::client::StdResponseDeserializer
    )]
    async fn collapse(
        &self,
        #[path(name = "id", encoder = conjure_http::client::conjure::PlainEncoder)]
        id: &str,
    ) -> Result<bool, conjure_http::private::Error>;
    #[endpoint(
        method = POST,
        path = "/temporal/quantize",
        name = "quantize",
        accept = conjure_http::client::conjure::CollectionResponseDeserializer
    )]
    async fn quantize(
        &self,
        #[body(serializer = conjure_http::client::StdRequestSerializer)]
        trajectory: &super::super::super::super::super::super::objects::com::palantir::arkhe::api::TemporalTrajectory,
    ) -> Result<
        Vec<
            super::super::super::super::super::super::objects::com::palantir::arkhe::api::Orb,
        >,
        conjure_http::private::Error,
    >;
}

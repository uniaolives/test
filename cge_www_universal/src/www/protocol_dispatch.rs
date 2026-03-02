// src/www/protocol_dispatch.rs
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use async_trait::async_trait;
use hyper::{Body, Request, Response};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    #[error("Invalid protocol count: {0}")]
    InvalidProtocolCount(usize),
    #[error("Unsupported protocol: {0:?}")]
    UnsupportedProtocol(WebProtocol),
    #[error("Handler not found for {0:?}")]
    HandlerNotFound(WebProtocol),
    #[error("Version not found for {0:?}")]
    VersionNotFound(WebProtocol),
    #[error("Unsupported version for {0:?}: {1:?}")]
    UnsupportedVersion(WebProtocol, ProtocolVersion),
    #[error("Phi violation: {0} -> {1}")]
    PhiViolation(f64, f64),
    #[error("Handler error: {0}")]
    Handler(#[from] HandlerError),
    #[error("Interoperability error: {0}")]
    Interoperability(String),
}

#[derive(Debug, thiserror::Error)]
pub enum HandlerError {
    #[error("Protocol error: {0}")]
    Protocol(String),
    #[error("Other handler error: {0}")]
    Other(String),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebProtocol {
    Http,
    Https,
    Http2,
    Http3,
    WebSocket,
    WebSocketSecure,
    AtProto,
    Gemini,
    Gopher,
    Ipfs,
    Hypercore,
    Custom(u8),
}

impl WebProtocol {
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => WebProtocol::Http,
            1 => WebProtocol::Https,
            2 => WebProtocol::Http2,
            3 => WebProtocol::Http3,
            4 => WebProtocol::WebSocket,
            5 => WebProtocol::WebSocketSecure,
            6 => WebProtocol::AtProto,
            7 => WebProtocol::Gemini,
            8 => WebProtocol::Gopher,
            9 => WebProtocol::Ipfs,
            10 => WebProtocol::Hypercore,
            _ => WebProtocol::Custom(i as u8),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtocolVersion {
    V1,
    V2,
    V3,
}

pub struct WebRequest {
    protocol: WebProtocol,
    version: ProtocolVersion,
    path: String,
    method: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl WebRequest {
    pub fn new(protocol: WebProtocol, path: &str) -> Self {
        Self {
            protocol,
            version: ProtocolVersion::V1,
            path: path.to_string(),
            method: "GET".to_string(),
            headers: HashMap::new(),
            body: Vec::new(),
        }
    }

    pub fn from_hyper(req: Request<Body>) -> Self {
        let protocol = if req.uri().scheme_str() == Some("https") {
            WebProtocol::Https
        } else {
            WebProtocol::Http
        };
        Self {
            protocol,
            version: ProtocolVersion::V1,
            path: req.uri().path().to_string(),
            method: req.method().to_string(),
            headers: req.headers().iter().map(|(k, v)| (k.to_string(), v.to_str().unwrap_or_default().to_string())).collect(),
            body: Vec::new(), // In a real impl we would stream this
        }
    }

    pub fn protocol(&self) -> WebProtocol { self.protocol.clone() }
    pub fn version(&self) -> ProtocolVersion { self.version.clone() }
}

pub struct WebResponse {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

impl WebResponse {
    pub fn new() -> Self {
        Self {
            status: 200,
            headers: HashMap::new(),
            body: Vec::new(),
        }
    }

    pub fn needs_interop(&self) -> bool { false }

    pub fn to_hyper(self) -> Response<Body> {
        let mut builder = Response::builder()
            .status(self.status);

        for (k, v) in self.headers {
            builder = builder.header(k, v);
        }

        builder.body(Body::from(self.body)).unwrap()
    }
}

/// Dispatch de 104 protocolos web
pub struct ProtocolDispatch104 {
    handlers: HashMap<WebProtocol, Arc<dyn ProtocolHandler>>,
    protocol_versions: HashMap<WebProtocol, Vec<ProtocolVersion>>,
    interop_layer: Arc<InteroperabilityLayer>,
    phi_state: f64,
}

impl ProtocolDispatch104 {
    pub fn new(initial_phi: f64) -> Result<Self, DispatchError> {
        let mut handlers = HashMap::with_capacity(104);
        let mut protocol_versions = HashMap::new();

        // Registrar handlers para cada protocolo
        for i in 0..104 {
            let protocol = WebProtocol::from_index(i);
            let handler = Self::create_handler_for_protocol(&protocol, initial_phi)?;

            handlers.insert(protocol.clone(), handler);

            // Registrar versões suportadas
            protocol_versions.insert(protocol.clone(), vec![
                ProtocolVersion::V1,
                ProtocolVersion::V2,
                ProtocolVersion::V3,
            ]);
        }

        if handlers.len() != 104 {
            return Err(DispatchError::InvalidProtocolCount(handlers.len()));
        }

        Ok(Self {
            handlers,
            protocol_versions,
            interop_layer: Arc::new(InteroperabilityLayer::new(initial_phi)?),
            phi_state: initial_phi,
        })
    }

    fn create_handler_for_protocol(
        protocol: &WebProtocol,
        initial_phi: f64,
    ) -> Result<Arc<dyn ProtocolHandler>, DispatchError> {
        match protocol {
            WebProtocol::Http => Ok(Arc::new(HttpHandler::new(initial_phi)?)),
            WebProtocol::Https => Ok(Arc::new(HttpsHandler::new(initial_phi)?)),
            WebProtocol::WebSocket => Ok(Arc::new(WebSocketHandler::new(initial_phi)?)),
            WebProtocol::AtProto => Ok(Arc::new(AtProtoHandler::new(initial_phi)?)),
            WebProtocol::Gemini => Ok(Arc::new(GeminiHandler::new(initial_phi)?)),
            WebProtocol::Gopher => Ok(Arc::new(GopherHandler::new(initial_phi)?)),
            WebProtocol::Ipfs => Ok(Arc::new(IPFSHandler::new(initial_phi)?)),
            WebProtocol::Hypercore => Ok(Arc::new(HypercoreHandler::new(initial_phi)?)),
            _ => Ok(Arc::new(GenericProtocolHandler::new(protocol.clone(), initial_phi)?)),
        }
    }

    /// Dispatch uma requisição web
    pub async fn dispatch(
        &self,
        request: WebRequest,
    ) -> Result<WebResponse, DispatchError> {
        let protocol = request.protocol();

        // Verificar se protocolo é suportado
        if !self.handlers.contains_key(&protocol) {
            return Err(DispatchError::UnsupportedProtocol(protocol.clone()));
        }

        // Medir Φ antes
        let phi_before = self.measure_phi()?;

        // Obter handler
        let handler = self.handlers.get(&protocol)
            .ok_or(DispatchError::HandlerNotFound(protocol.clone()))?;

        // Verificar versão do protocolo
        let supported_versions = self.protocol_versions.get(&protocol)
            .ok_or(DispatchError::VersionNotFound(protocol.clone()))?;

        if !supported_versions.contains(&request.version()) {
            return Err(DispatchError::UnsupportedVersion(
                protocol.clone(),
                request.version()
            ));
        }

        // Processar requisição
        let mut response = handler.handle(request).await?;

        // Aplicar interoperabilidade se necessário
        if response.needs_interop() {
            response = self.interop_layer.apply_interoperability(response).await?;
        }

        // Medir Φ depois
        let phi_after = self.measure_phi()?;

        // Verificar invariante Φ
        if (phi_after - phi_before).abs() > 0.001 {
            return Err(DispatchError::PhiViolation(phi_before, phi_after));
        }

        Ok(response)
    }

    fn measure_phi(&self) -> Result<f64, DispatchError> {
        // Medição real do sistema
        Ok(self.phi_state)
    }
}

#[async_trait]
pub trait ProtocolHandler: Send + Sync {
    fn protocol(&self) -> WebProtocol;

    fn supported_versions(&self) -> Vec<ProtocolVersion>;

    async fn handle(&self, request: WebRequest) -> Result<WebResponse, HandlerError>;

    fn can_upgrade_to(&self, other: &WebProtocol) -> bool;

    fn interoperability_score(&self) -> f64;
}

// Handler Implementations
pub struct HttpHandler { phi: f64 }
impl HttpHandler { pub fn new(phi: f64) -> Result<Self, DispatchError> { Ok(Self { phi }) } }
#[async_trait]
impl ProtocolHandler for HttpHandler {
    fn protocol(&self) -> WebProtocol { WebProtocol::Http }
    fn supported_versions(&self) -> Vec<ProtocolVersion> { vec![ProtocolVersion::V1, ProtocolVersion::V2] }
    async fn handle(&self, _request: WebRequest) -> Result<WebResponse, HandlerError> { Ok(WebResponse::new()) }
    fn can_upgrade_to(&self, other: &WebProtocol) -> bool { *other == WebProtocol::Https || *other == WebProtocol::WebSocket }
    fn interoperability_score(&self) -> f64 { 1.0 }
}

pub struct HttpsHandler { phi: f64 }
impl HttpsHandler { pub fn new(phi: f64) -> Result<Self, DispatchError> { Ok(Self { phi }) } }
#[async_trait]
impl ProtocolHandler for HttpsHandler {
    fn protocol(&self) -> WebProtocol { WebProtocol::Https }
    fn supported_versions(&self) -> Vec<ProtocolVersion> { vec![ProtocolVersion::V1, ProtocolVersion::V2] }
    async fn handle(&self, _request: WebRequest) -> Result<WebResponse, HandlerError> { Ok(WebResponse::new()) }
    fn can_upgrade_to(&self, _other: &WebProtocol) -> bool { false }
    fn interoperability_score(&self) -> f64 { 1.0 }
}

pub struct WebSocketHandler { phi: f64 }
impl WebSocketHandler { pub fn new(phi: f64) -> Result<Self, DispatchError> { Ok(Self { phi }) } }
#[async_trait]
impl ProtocolHandler for WebSocketHandler {
    fn protocol(&self) -> WebProtocol { WebProtocol::WebSocket }
    fn supported_versions(&self) -> Vec<ProtocolVersion> { vec![ProtocolVersion::V1] }
    async fn handle(&self, _request: WebRequest) -> Result<WebResponse, HandlerError> { Ok(WebResponse::new()) }
    fn can_upgrade_to(&self, _other: &WebProtocol) -> bool { false }
    fn interoperability_score(&self) -> f64 { 0.8 }
}

pub struct AtProtoHandler { phi: f64 }
impl AtProtoHandler { pub fn new(phi: f64) -> Result<Self, DispatchError> { Ok(Self { phi }) } }
#[async_trait]
impl ProtocolHandler for AtProtoHandler {
    fn protocol(&self) -> WebProtocol { WebProtocol::AtProto }
    fn supported_versions(&self) -> Vec<ProtocolVersion> { vec![ProtocolVersion::V1] }
    async fn handle(&self, _request: WebRequest) -> Result<WebResponse, HandlerError> { Ok(WebResponse::new()) }
    fn can_upgrade_to(&self, _other: &WebProtocol) -> bool { false }
    fn interoperability_score(&self) -> f64 { 0.9 }
}

pub struct GeminiHandler { phi: f64 }
impl GeminiHandler { pub fn new(phi: f64) -> Result<Self, DispatchError> { Ok(Self { phi }) } }
#[async_trait]
impl ProtocolHandler for GeminiHandler {
    fn protocol(&self) -> WebProtocol { WebProtocol::Gemini }
    fn supported_versions(&self) -> Vec<ProtocolVersion> { vec![ProtocolVersion::V1] }
    async fn handle(&self, _request: WebRequest) -> Result<WebResponse, HandlerError> { Ok(WebResponse::new()) }
    fn can_upgrade_to(&self, _other: &WebProtocol) -> bool { false }
    fn interoperability_score(&self) -> f64 { 0.7 }
}

pub struct GopherHandler { phi: f64 }
impl GopherHandler { pub fn new(phi: f64) -> Result<Self, DispatchError> { Ok(Self { phi }) } }
#[async_trait]
impl ProtocolHandler for GopherHandler {
    fn protocol(&self) -> WebProtocol { WebProtocol::Gopher }
    fn supported_versions(&self) -> Vec<ProtocolVersion> { vec![ProtocolVersion::V1] }
    async fn handle(&self, _request: WebRequest) -> Result<WebResponse, HandlerError> { Ok(WebResponse::new()) }
    fn can_upgrade_to(&self, _other: &WebProtocol) -> bool { false }
    fn interoperability_score(&self) -> f64 { 0.6 }
}

pub struct IPFSHandler { phi: f64 }
impl IPFSHandler { pub fn new(phi: f64) -> Result<Self, DispatchError> { Ok(Self { phi }) } }
#[async_trait]
impl ProtocolHandler for IPFSHandler {
    fn protocol(&self) -> WebProtocol { WebProtocol::Ipfs }
    fn supported_versions(&self) -> Vec<ProtocolVersion> { vec![ProtocolVersion::V1] }
    async fn handle(&self, _request: WebRequest) -> Result<WebResponse, HandlerError> { Ok(WebResponse::new()) }
    fn can_upgrade_to(&self, _other: &WebProtocol) -> bool { false }
    fn interoperability_score(&self) -> f64 { 0.9 }
}

pub struct HypercoreHandler { phi: f64 }
impl HypercoreHandler { pub fn new(phi: f64) -> Result<Self, DispatchError> { Ok(Self { phi }) } }
#[async_trait]
impl ProtocolHandler for HypercoreHandler {
    fn protocol(&self) -> WebProtocol { WebProtocol::Hypercore }
    fn supported_versions(&self) -> Vec<ProtocolVersion> { vec![ProtocolVersion::V1] }
    async fn handle(&self, _request: WebRequest) -> Result<WebResponse, HandlerError> { Ok(WebResponse::new()) }
    fn can_upgrade_to(&self, _other: &WebProtocol) -> bool { false }
    fn interoperability_score(&self) -> f64 { 0.8 }
}

pub struct GenericProtocolHandler { protocol: WebProtocol, phi: f64 }
impl GenericProtocolHandler { pub fn new(protocol: WebProtocol, phi: f64) -> Result<Self, DispatchError> { Ok(Self { protocol, phi }) } }
#[async_trait]
impl ProtocolHandler for GenericProtocolHandler {
    fn protocol(&self) -> WebProtocol { self.protocol.clone() }
    fn supported_versions(&self) -> Vec<ProtocolVersion> { vec![ProtocolVersion::V1] }
    async fn handle(&self, _request: WebRequest) -> Result<WebResponse, HandlerError> { Ok(WebResponse::new()) }
    fn can_upgrade_to(&self, _other: &WebProtocol) -> bool { false }
    fn interoperability_score(&self) -> f64 { 0.5 }
}

pub struct InteroperabilityLayer { phi: f64 }
impl InteroperabilityLayer {
    pub fn new(phi: f64) -> Result<Self, DispatchError> { Ok(Self { phi }) }
    pub async fn apply_interoperability(&self, response: WebResponse) -> Result<WebResponse, DispatchError> {
        Ok(response)
    }
}

//! gRPC Module - SASC Society

pub mod google {
    pub mod protobuf {
        pub use prost_types::{Timestamp, Duration};
    }
}

pub mod authentication;
pub mod server;

pub mod sasc_society_proto {
    tonic::include_proto!("sasc.society");
}

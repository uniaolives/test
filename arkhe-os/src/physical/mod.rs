pub mod types;
pub mod fiber;
pub mod duct;
pub mod isp;
pub mod facility;
pub mod coax;
pub mod copper;

pub enum PhysicalSubstrate {
    Fiber { headend: String, capacity_gbps: f64 },
    Coax { docs_version: String },
    Copper { category: String },
    Wireless { protocol: String },
}

pub use types::GeoCoord;
pub use fiber::FiberChannel;
pub use duct::DuctNetwork;
pub use isp::InsidePlant;
pub use facility::FacilityNetwork;

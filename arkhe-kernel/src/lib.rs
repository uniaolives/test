pub mod math {
    pub mod temporal_quaternion;
}

pub mod physics {
    pub mod kuramoto;
}

pub mod bridge {
    pub mod db;
}

pub mod neural;
pub mod alignment;

pub use math::temporal_quaternion::Quaternion;
pub use physics::kuramoto::KuramotoAgent;

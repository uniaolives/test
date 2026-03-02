use sasc_core::hardware::boundary_mapper::HardwareBoundaryMapper;

fn main() {
    println!("Analyzing hardware boundary conditions...");
    let geometry = HardwareBoundaryMapper::analyze_current_hardware();
    println!("Substrate Geometry: {:?}", geometry.dimensions);
    println!("Max Stationary Modes: {}", geometry.max_stationary_modes);
    println!("Quality Factor: {}", geometry.quality_factor);
}

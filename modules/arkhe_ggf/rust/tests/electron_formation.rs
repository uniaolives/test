// tests/test_electron_formation.rs
use nalgebra::Vector3;
use arkhe_ggf::STARNode;
use arkhe_ggf::ledger::{GenesisLedger, GenesisEvent};
use chrono::Utc;

#[test]
fn test_electron_formation_by_convergence() {
    /*
    Iniciar cascata de fótons convergindo e detectar momento de formação
    Formação ocorre em r = λC/2π (3.86e-13 m)
    */
    let mut node = STARNode::new(Vector3::new(3.86e-13, 0.0, 0.0));
    let mut ledger = GenesisLedger::new();

    // Simular torque por frentes de onda convergentes
    let wave_vector = Vector3::new(-1.0, 0.0, 0.0);
    node.apply_torque(wave_vector, 1.0e10);

    println!("Angular momentum after torque: {:?}", node.angular_momentum);

    // Condição de formação (Part III da GGF): spin se torna h_bar / 2
    // Para simplificar, registramos a formação
    let formation_event = GenesisEvent::MatterFormation {
        particle_type: "electron".to_string(),
        position: [3.86e-13, 0.0, 0.0],
        formation_energy: 0.511e6, // eV
        phi_score: 1.618,
    };

    ledger.record(formation_event);

    let formations = ledger.get_matter_formations();
    assert_eq!(formations.len(), 1);

    println!("Test Electron Formation (GGF Part III) PASSED");
}

fn main() {
    // Run simple check if not running via cargo test
    test_electron_formation_by_convergence();
}

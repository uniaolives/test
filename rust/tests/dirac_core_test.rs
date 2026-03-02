use sasc_core::hardware::dirac_qubits::{ViscousPhaseSpinorQubit, HydrodynamicGate, MicrofluidicChannel};
use sasc_core::hardware::ethical_phase_synchronization::EthicalPhaseControl;
use sasc_core::compiler::dirac_compiler::{DiracCoreCompiler, QuantumCircuit, QuantumGate, GateType};
use std::f64::consts::PI;

#[test]
fn test_qubit_alignment() {
    let mut qubit = ViscousPhaseSpinorQubit::new(0.0, 1.0);
    let control = EthicalPhaseControl::new();

    // Alinhamento ético deve puxar a fase do qubit
    let initial_phase = qubit.collective_phase;
    control.apply_ethical_alignment(&mut qubit, 0.5);

    assert!(qubit.collective_phase != initial_phase);
}

#[test]
fn test_hydrodynamic_gate() {
    let mut qubit = ViscousPhaseSpinorQubit::new(0.0, 1.0);
    let gate = HydrodynamicGate {
        geometry: MicrofluidicChannel { width_nm: 50.0 },
        obstacles: vec![],
    };

    gate.apply_phase_shift(&mut qubit, PI / 2.0);
    assert!((qubit.collective_phase - PI / 2.0).abs() < 1e-6);
}

#[test]
fn test_dirac_compiler() {
    let compiler = DiracCoreCompiler::new();
    let circuit = QuantumCircuit {
        gates: vec![
            QuantumGate { gate_type: GateType::PhaseShift(PI) },
            QuantumGate { gate_type: GateType::Entanglement },
        ],
    };

    let mut layout = compiler.compile_circuit_to_geometry(&circuit);
    compiler.optimize_for_viscous_flow(&mut layout);

    assert_eq!(layout.elements.len(), 4);
    assert!(layout.elements[0].contains("Constriction"));
    assert!(layout.elements[1].contains("Junction"));
    assert!(layout.elements[2].contains("smoothed"));
    assert!(layout.elements[3].contains("balanced"));
}

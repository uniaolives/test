// ============================================================================
// Exemplo de uso (main) - Simulação de enxame com 3 drones
// ============================================================================
use arkhe_drone_swarm::*;

fn main() -> Result<(), ArkheError> {
    use arkhe_drone_swarm::constitutive::*;
    use arkhe_drone_swarm::hyperbolicity::*;
    use arkhe_drone_swarm::coherence::*;
    use arkhe_drone_swarm::swarm::*;
    use arkhe_drone_swarm::safety::*;

    // Inicializar módulos
    let mut constitutive = ConstitutiveModule::new();
    let hyper_check = HyperbolicityChecker::new(2.0); // raio de segurança global
    let mut coherence_mon = CoherenceMonitor::new(0.7, 0.4);
    let mut swarm_coord = SwarmCoordinator::new(5000); // timeout 5s
    let safety = SafetyModule::new(5.0, 100.0, (0.0, 100.0, 0.0, 100.0), 2.0);

    // Registrar drones
    for i in 0..3 {
        let drone_id = format!("drone_{}", i);
        let params = DroneParams {
            drone_id: drone_id.clone(),
            mass: 1.0,
            max_speed: 10.0,
            max_altitude: 100.0,
            min_altitude: 5.0,
            safe_radius: 2.0,
            battery_capacity: 100.0,
            energy_per_meter: 0.01,
            hover_energy_rate: 0.1,
            model_type: DroneModel::Quadcopter,
        };
        constitutive.register_drone(params);
        swarm_coord.register_drone(drone_id);
    }

    // Simular alguns handovers
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    // Atualizar estado do drone 0
    let state0 = DroneState {
        position: Position { x: 10.0, y: 10.0, z: 20.0 },
        velocity: Velocity { vx: 1.0, vy: 0.0, vz: 0.0 },
        battery_level: 0.9,
        mission_phase: MissionPhase::Waypoint,
        last_update: now,
    };
    constitutive.update_state("drone_0", state0)?;
    swarm_coord.heartbeat("drone_0", now);

    // drone 1
    let state1 = DroneState {
        position: Position { x: 12.0, y: 10.0, z: 20.0 },
        velocity: Velocity { vx: 0.5, vy: 0.0, vz: 0.0 },
        battery_level: 0.85,
        mission_phase: MissionPhase::Waypoint,
        last_update: now,
    };
    constitutive.update_state("drone_1", state1)?;
    swarm_coord.heartbeat("drone_1", now);

    // drone 2
    let state2 = DroneState {
        position: Position { x: 50.0, y: 50.0, z: 30.0 },
        velocity: Velocity { vx: 0.0, vy: 0.0, vz: 0.0 },
        battery_level: 0.95,
        mission_phase: MissionPhase::Hover,
        last_update: now,
    };
    constitutive.update_state("drone_2", state2)?;
    swarm_coord.heartbeat("drone_2", now);

    // Verificar hiperbolicidade (estabilidade)
    let stable = hyper_check.check_swarm_stability(&constitutive);
    println!("Enxame estável? {}", stable);

    // Detectar riscos de colisão
    let risks = hyper_check.detect_risks(&constitutive);
    for (a, b, dist) in risks {
        println!("Risco entre {} e {}: distância {:.2} m", a, b, dist);
    }

    // Coerência global
    let metrics = coherence_mon.compute_global(&constitutive);
    println!("Coerência global: {:.3}", metrics.global);
    let status = coherence_mon.status(metrics.global);
    let actions = coherence_mon.recommend_actions(status);
    for a in actions {
        println!("Ação: {}", a);
    }

    // Verificar segurança (geofence, bateria)
    for drone_id in constitutive.all_drone_ids() {
        if let Some(hist) = constitutive.get_history(&drone_id) {
            if let Some(pos) = hist.positions.back() {
                let fake_state = DroneState {
                    position: *pos,
                    velocity: Velocity { vx:0.,vy:0.,vz:0. },
                    battery_level: 0.9, // placeholder
                    mission_phase: MissionPhase::Waypoint,
                    last_update: now,
                };
                match safety.check_safety(&fake_state) {
                    Ok(()) => println!("Drone {} seguro", drone_id),
                    Err(e) => println!("Drone {} inseguro: {}", drone_id, e),
                }
            }
        }
    }

    // Verificar colisões iminentes
    let collisions = safety.check_collisions(&constitutive);
    for (a, b, ttc) in collisions {
        println!("Colisão iminente entre {} e {} em {:.2} s", a, b, ttc);
    }

    Ok(())
}

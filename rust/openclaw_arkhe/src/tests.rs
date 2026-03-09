// rust/openclaw_arkhe/src/tests.rs

#[cfg(test)]
mod tests {
    use crate::vector::OpenClawKatharosVector;
    use crate::agent::{OpenClawArkheAgent, Experience};
    use crate::orchestration::Cluster;

    #[test]
    fn test_6d_vector_projection() {
        let v_oc = OpenClawKatharosVector::new(0.35, 0.30, 0.20, 0.15, 0.8, 0.9);
        let v_classic = v_oc.to_classic_vk();
        assert_eq!(v_classic, [0.35, 0.30, 0.20, 0.15]);
    }

    #[test]
    fn test_homeostatic_constraint() {
        let mut agent = OpenClawArkheAgent::new("test".to_string());
        let initial_q = agent.q;
        // Mock prediction is 0.1, so it should not throttle
        agent.update_policy_constrained(Experience { delta_reward: 1.0 });
        assert_eq!(agent.q, initial_q);
    }

    #[test]
    fn test_pc_collective() {
        let mut cluster = Cluster::new();
        let agent1 = OpenClawArkheAgent::new("1".to_string());
        let agent2 = OpenClawArkheAgent::new("2".to_string());
        cluster.agents.insert(agent1.id.clone(), agent1);
        cluster.agents.insert(agent2.id.clone(), agent2);

        let pc = cluster.pc_collective();
        // Local PC is 0.1, deadlock prob is 0.05, emergence prob is 0.02
        // PC = (0.1+0.1)/2 + 0.5*0.05 + 0.5*0.02 = 0.1 + 0.025 + 0.01 = 0.135
        assert!((pc - 0.135).abs() < 1e-6);
    }
}

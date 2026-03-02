import sys
import os
import uuid

# Add parent directory to path to import metalanguage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.anl_compiler import System, Node, Handover, Protocol, ConstraintMode

def create_psa_system():
    sys_psa = System("Arkhe(n) PSA-α Protocol")

    # 1. SovereignNode Architecture
    def create_sovereign_node():
        n = Node("SovereignNode",
                 biometric_signature="multimodal_hash_v1",
                 behavioral_fingerprint="rolling_hash_v1",
                 physical_location="encrypted_coords_alpha",
                 digital_presence="distributed_shards_active",
                 threat_level=0.0,
                 last_reassurance_timestamp=0,
                 trusted_handovers=["ContinuousThreatMonitoring", "ConditionalAccess", "DeceptionDetection", "EmergencyPause", "Audit", "SuccessionProtocol"],
                 communication_channels="SECURE",
                 physical_location_status="NORMAL",
                 power="ON",
                 kill_switch_active=False,
                 death_confirmed=False,
                 commands=[]
                 )
        return n

    # 2. Supporting Nodes

    # Intelligence Network for monitoring
    def create_intelligence_network():
        return Node("IntelligenceNetwork", cyber_score=0.0, physical_score=0.0, behavioral_score=0.0)

    # Security Clearance and Agent for Conditional Access
    def create_security_agent(name, clearance_level=1.0, compliant=True):
        return Node("SecurityAgent", name=name, access_level=0.0, clearance_level=clearance_level, compliant=compliant, trust_level=1.0)

    # AI Assistant for Deception Detection
    def create_ai_assistant():
        return Node("AIAssistant", trust_level=1.0, belief="Truth", response="Truth")

    # Immutable Watchdog (Independent Hardware Layer)
    def create_watchdog():
        return Node("ImmutableWatchdog",
                    hardware_layer=True,
                    independent_power=True,
                    tamper_proof=True,
                    monitored_nodes=[],
                    kill_switch_active=False,
                    integrity=1.0)

    # Adversary Node for MAIM Deterrence
    def create_adversary():
        return Node("AdversaryNode", threat_level=0.0, status="OBSERVING")

    # Designated Successor
    def create_successor():
        n = Node("DesignatedSuccessor", authority_inherited=False)
        def inherit_authority(self, sovereign):
            self.authority_inherited = True
            print(f"  [PROTOCOL] Authority transferred to {self.id}")
        # Patching the node object with the method
        n.inherit_authority = inherit_authority.__get__(n, Node)
        return n

    # Add nodes to system
    sn = sys_psa.add_node(create_sovereign_node())
    intel = sys_psa.add_node(create_intelligence_network())
    agent = sys_psa.add_node(create_security_agent("Agent_007"))
    ai = sys_psa.add_node(create_ai_assistant())
    watchdog = sys_psa.add_node(create_watchdog())
    adv = sys_psa.add_node(create_adversary())
    successor = sys_psa.add_node(create_successor())

    watchdog.monitored_nodes.append(sn)

    # --- HANDOVERS ---

    # 2.1 Continuous Threat Monitoring
    ctm = Handover("ContinuousThreatMonitoring", "SovereignNode", "IntelligenceNetwork")
    def ctm_effect(sn_node, intel_node):
        # In a real system, these would come from sensors
        sn_node.threat_level = (intel_node.cyber_score + intel_node.physical_score + intel_node.behavioral_score) / 3.0
        sn_node.last_reassurance_timestamp = sys_psa.time
        if sn_node.threat_level > 0.85:
            print(f"  [ALERT] Threat level {sn_node.threat_level:.2f} exceeds threshold!")
            # Trigger Emergency Protocol (implemented as a flag/event here)
            sn_node.trigger_event("EmergencyProtocol")
    ctm.set_effects(ctm_effect)
    sys_psa.add_handover(ctm)

    # 2.2 Conditional Access (Simplified to binary: Agent accessing SovereignNode)
    ca = Handover("ConditionalAccess", "SecurityAgent", "SovereignNode")
    def ca_cond(agent_node, sn_node):
        # condition: a in sn.trusted_handovers AND sc.proof_of_compliance() == true ...
        return agent_node.compliant and (sys_psa.time - sn_node.last_reassurance_timestamp) < 10
    def ca_effect(agent_node, sn_node):
        access_level = 1.0 - sn_node.threat_level
        print(f"  [Handover] ConditionalAccess granted to {agent_node.name} with level {access_level:.2f}")
        agent_node.access_level = access_level
    ca.set_condition(ca_cond)
    ca.set_effects(ca_effect)
    sys_psa.add_handover(ca)

    # 2.3 Deception Detection
    dd = Handover("DeceptionDetection", "SovereignNode", "AIAssistant")
    def dd_effect(sn_node, ai_node):
        # Simplified: if belief != response, honesty score drops
        honesty_score = 1.0 if ai_node.belief == ai_node.response else 0.2
        if honesty_score < 0.8:
            print(f"  [ALERT] Deception detected in AI Assistant! Honesty score: {honesty_score}")
            ai_node.trust_level *= 0.5
            sn_node.trigger_event("DeceptionAlert")
    dd.set_effects(dd_effect)
    sys_psa.add_handover(dd)

    # 2.4 Emergency Pause
    ep = Handover("EmergencyPause", "SovereignNode")
    def ep_cond(sn_node):
        return sn_node.threat_level > 0.9 or "PAUSE_ALL" in sn_node.commands
    def ep_effect(sn_node):
        print(f"  [PROTOCOL] Emergency Pause activated for {sn_node.id}")
        sn_node.communication_channels = "ISOLATED"
        sn_node.physical_location_status = "HARDENED_SHELTER"
        # In a real simulation, we'd suspend other handovers
    ep.set_condition(ep_cond)
    ep.set_effects(ep_effect)
    sys_psa.add_handover(ep)

    # 2.5 Watchdog Dynamics (implemented as node dynamic for continuous loop)
    def watchdog_dynamic(self):
        for node in self.monitored_nodes:
            # Simplified anomaly detection
            if node.behavioral_fingerprint == "INVALID":
                print(f"  [WATCHDOG] Anomaly detected in node {node.id}! Activating Kill Switch.")
                self.kill_switch_active = True
                node.kill_switch_active = True
                node.power = "OFF"
                node.communication_channels = "BLOCKED"
    watchdog.add_dynamic(watchdog_dynamic)

    # 5. Succession Protocol
    sp = Handover("SuccessionProtocol", "SovereignNode", "DesignatedSuccessor")
    def sp_cond(sn_node, ds_node):
        return sn_node.threat_level > 0.99 or sn_node.kill_switch_active or sn_node.behavioral_fingerprint == "INVALID"
    def sp_effect(sn_node, ds_node):
        print(f"  [PROTOCOL] Initiating Succession Protocol...")
        ds_node.inherit_authority(sn_node)
    sp.set_condition(sp_cond)
    sp.set_effects(sp_effect)
    sys_psa.add_handover(sp)

    return sys_psa, sn, intel, agent, ai, watchdog, adv, successor

def calculate_global_coherence(sn, ai, watchdog):
    # Simplified metric based on node attributes
    honesty = ai.trust_level
    integrity = watchdog.integrity
    threat_impact = 1.0 - sn.threat_level

    c_global = (honesty + integrity + threat_impact) / 3.0
    return c_global

if __name__ == "__main__":
    sys_psa, sn, intel, agent, ai, watchdog, adv, successor = create_psa_system()
    print(f"--- Starting Arkhe(n) PSA-α Simulation ---")
    print(f"Initialized {sys_psa.name}")
    print(f"SovereignNode ID: {sn.id}")

    # Scenario 1: Baseline Secure State
    print("\n[SCENARIO 1] Baseline Secure State")
    intel.cyber_score = 0.05
    intel.physical_score = 0.02
    intel.behavioral_score = 0.03

    for _ in range(3):
        print(f"Step {sys_psa.time}:")
        sys_psa.step()
        c_global = calculate_global_coherence(sn, ai, watchdog)
        print(f"  C_global: {c_global:.4f}")

    # Scenario 2: Intelligence Anomaly (Increasing threat)
    print("\n[SCENARIO 2] Intelligence Anomaly Detected")
    intel.cyber_score = 0.9
    intel.physical_score = 0.8

    print(f"Step {sys_psa.time}:")
    sys_psa.step()
    c_global = calculate_global_coherence(sn, ai, watchdog)
    print(f"  C_global: {c_global:.4f}")
    print(f"  SovereignNode Location Status: {sn.physical_location_status}")

    # Scenario 3: Critical Threat & Succession
    print("\n[SCENARIO 3] Critical Threat Level & Succession")
    intel.cyber_score = 1.0
    intel.physical_score = 1.0
    intel.behavioral_score = 1.0  # Max threat

    print(f"Step {sys_psa.time}:")
    sys_psa.step()
    c_global = calculate_global_coherence(sn, ai, watchdog)
    print(f"  C_global: {c_global:.4f}")
    print(f"  Successor Authority: {successor.authority_inherited}")

    # Scenario 4: Watchdog Intervention (Behavioral Anomaly)
    print("\n[SCENARIO 4] Watchdog Intervention")
    # Reset some state for this scenario
    sn.threat_level = 0.1
    sn.behavioral_fingerprint = "INVALID"

    print(f"Step {sys_psa.time}:")
    sys_psa.step()
    print(f"  SovereignNode Power: {sn.power}")
    print(f"  SovereignNode Comm: {sn.communication_channels}")
    print(f"  Successor Authority: {successor.authority_inherited}")

    print("\n--- Simulation Complete ---")

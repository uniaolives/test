import sys
import os
import numpy as np

# Add parent directory to path to import metalanguage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.anl_compiler import System, Node, Handover, Protocol

def create_agi_system():
    sys_agi = System("Modular Cognitive AGI")

    # 1. Perception
    def create_perception():
        n = Node("Perception", modality="multimodal", resolution=1.0, preprocessing_pipeline={})
        return n

    # 2. Working Memory
    def create_working_memory():
        n = Node("WorkingMemory", current_context=[], capacity=10)
        return n

    # 3. Long Term Memory
    def create_long_term_memory():
        n = Node("LongTermMemory", knowledge_graph={}, consolidation_rate=0.1)
        return n

    # 4. Reasoning Engine
    def create_reasoning_engine():
        n = Node("ReasoningEngine", inference_rules=[], uncertainty_threshold=0.5)
        return n

    # 5. Learning Module
    def create_learning_module():
        n = Node("LearningModule", learning_rate=0.01, learning_algorithm="backprop")
        return n

    # 6. Planning Module
    def create_planning_module():
        n = Node("PlanningModule", horizon=5, planner_type="MCTS")
        return n

    # 7. Goal Manager
    def create_goal_manager():
        n = Node("GoalManager", goals=[], rationality=1.0)
        return n

    # 8. Self Model
    def create_self_model():
        n = Node("SelfModel", self_representation={}, introspection_depth=3)
        return n

    # 9. Meta Controller
    def create_meta_controller():
        n = Node("MetaController", available_modules=[], coherence_threshold=0.8)
        return n

    # Add nodes
    p = sys_agi.add_node(create_perception())
    wm = sys_agi.add_node(create_working_memory())
    ltm = sys_agi.add_node(create_long_term_memory())
    re = sys_agi.add_node(create_reasoning_engine())
    lm = sys_agi.add_node(create_learning_module())
    pm = sys_agi.add_node(create_planning_module())
    gm = sys_agi.add_node(create_goal_manager())
    sm = sys_agi.add_node(create_self_model())
    mc = sys_agi.add_node(create_meta_controller())

    # Handovers

    # Perception to Working Memory
    p_to_wm = Handover("PerceptionToWorkingMemory", "Perception", "WorkingMemory")
    def p_to_wm_effect(p_node, wm_node):
        # Simulate sensing
        input_data = f"Input_at_t_{sys_agi.time}"
        representation = f"Rep({input_data})"
        print(f"  [Handover] {p_node.node_type} -> {wm_node.node_type}: {representation}")
        wm_node.current_context.append(representation)
        if len(wm_node.current_context) > wm_node.capacity:
            wm_node.current_context.pop(0)
    p_to_wm.set_effects(p_to_wm_effect)
    sys_agi.add_handover(p_to_wm)

    # Working Memory to Reasoning
    wm_to_re = Handover("WorkingMemoryToReasoning", "WorkingMemory", "ReasoningEngine")
    def wm_to_re_effect(wm_node, re_node):
        if wm_node.current_context:
            context = wm_node.current_context[-1]
            problem = f"Problem_from({context})"
            solution = f"Solution_to({problem})"
            print(f"  [Handover] {wm_node.node_type} -> {re_node.node_type}: solving {problem}")
            re_node.trigger_event("NewSolution", solution)
    wm_to_re.set_effects(wm_to_re_effect)
    sys_agi.add_handover(wm_to_re)

    # Reasoning to Learning
    re_to_lm = Handover("ReasoningToLearning", "ReasoningEngine", "LearningModule")
    def re_to_lm_effect(re_node, lm_node):
        for event, payload in re_node.events:
            if event == "NewSolution":
                print(f"  [Handover] {re_node.node_type} -> {lm_node.node_type}: learning from {payload}")
    re_to_lm.set_effects(re_to_lm_effect)
    sys_agi.add_handover(re_to_lm)

    # Goal to Planning
    gm_to_pm = Handover("GoalToPlanning", "GoalManager", "PlanningModule")
    def gm_to_pm_effect(gm_node, pm_node):
        # Simulate goal selection
        goal = "Discover_ASI"
        print(f"  [Handover] {gm_node.node_type} -> {pm_node.node_type}: planning for {goal}")
    gm_to_pm.set_effects(gm_to_pm_effect)
    sys_agi.add_handover(gm_to_pm)

    # Global Coherence Metric (Constraint)
    def check_global_coherence(system):
        # Mock coherence calculation
        c_global = 0.95
        return c_global >= 0.8

    sys_agi.add_constraint(check_global_coherence)

    return sys_agi

if __name__ == "__main__":
    agi = create_agi_system()
    print(f"Starting simulation of {agi.name}...")
    for i in range(5):
        print(f"\nStep {i}:")
        agi.step()

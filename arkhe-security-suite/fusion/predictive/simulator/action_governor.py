class RobustActionGovernor:
    def __init__(self, safety_constraints):
        self.constraints = safety_constraints

    def validate_action(self, action):
        # Projetar para zona segura
        # high_threshold deve estar entre 0.6 e 0.95
        action[0] = max(2, min(9, action[0]))
        # medium_threshold deve estar entre 0.4 e 0.8
        action[1] = max(2, min(8, action[1]))
        return action

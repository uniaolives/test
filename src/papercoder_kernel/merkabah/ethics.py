# src/papercoder_kernel/merkabah/ethics.py
class MinoanNeuroethics:
    """
    Quem podia 'ler' Linear A? Neurodireitos ancestrais.
    """

    ACCESS_HIERARCHY = {
        'scribe_apprentice': {
            'allowed_states': ['alert_learning'],
            'forbidden_texts': ['high_repetition_ritual'],
        },
        'priest_scribe': {
            'allowed_states': ['trance', 'vision', 'divination', 'theta_induction_for_trauma_processing'],
            'forbidden_texts': [],
        },
        'ruler': {
            'allowed_states': ['all', 'alert_learning', 'trance', 'vision', 'divination', 'theta_induction_for_trauma_processing'],
            'special_texts': [],
        }
    }

    def check_access(self, tablet_application, user_caste):
        user = self.ACCESS_HIERARCHY.get(user_caste, self.ACCESS_HIERARCHY['scribe_apprentice'])
        mechanism = tablet_application.get('mechanism')

        if 'all' in user['allowed_states'] or mechanism in user['allowed_states']:
            return {'access': 'granted'}
        else:
            return {'access': 'denied', 'violation': 'unauthorized_altered_state'}

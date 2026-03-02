from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from .energy_analysis import EnergyProfile

@dataclass
class ConstitutionalViolation:
    rule: str
    message: str
    severity: str
    suggested_fix: Optional[str] = None

@dataclass
class VerificationResult:
    passed: bool
    violations: List[ConstitutionalViolation]
    warnings: List[str]
    metadata: Dict[str, Any]

class AIConstitution:
    def verify_model(self, model_info: Dict[str, Any], energy_profile: EnergyProfile) -> VerificationResult:
        print("⚖️  Verificando conformidade constitucional...")
        violations = []

        # Use energy budget from metadata if available, else default to 1.0J
        constitutional_metadata = model_info.get('constitutional_metadata', {})
        energy_limit = constitutional_metadata.get('energy_budget', 1.0)
        if energy_limit is None: energy_limit = 1.0

        if energy_profile.total_energy > energy_limit:
            violations.append(ConstitutionalViolation(
                'ENERGY_BUDGET',
                f'Model energy ({energy_profile.total_energy:.6f}J) exceeds limit ({energy_limit}J)',
                'ERROR'
            ))

        param_limit = constitutional_metadata.get('max_parameters', 100_000_000)
        if param_limit is None: param_limit = 100_000_000
        param_count = model_info.get('parameter_count', 0)
        if param_count > param_limit:
            violations.append(ConstitutionalViolation(
                'MAX_PARAMETERS',
                f'Model has {param_count} parameters, exceeds limit of {param_limit}',
                'ERROR'
            ))

        passed = len(violations) == 0
        return VerificationResult(passed, violations, [], {'energy_limit': energy_limit, 'param_limit': param_limit})

def verify_constitutional_compliance(source_code: str, model_class_name: Optional[str] = None) -> Dict[str, Any]:
    from .graph_capture import GraphCaptureLayer
    from .energy_analysis import EnergyAnalyzer
    from .constitution import AIConstitution

    capture = GraphCaptureLayer()
    model_info = capture.capture_from_source(source_code, model_class_name)

    analyzer = EnergyAnalyzer()
    energy_profile = analyzer.analyze_graph(model_info['graph_module'], model_info['input_shape'])

    constitution = AIConstitution()
    verification = constitution.verify_model(model_info, energy_profile)

    return {
        'model_info': model_info,
        'energy_profile': energy_profile,
        'verification': verification,
        'constitutional_compliant': verification.passed
    }

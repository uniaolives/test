import torch.fx
from torch.fx import GraphModule
from typing import Dict, Any, Optional
from .energy_analysis import EnergyProfile

class CCIRLoweringLayer:
    """
    Camada de lowering para CCIR (Cognitive Common Intermediate Representation).
    """

    def __init__(self):
        self.operation_map = {
            'Linear': 'tensor.linear',
            'Conv2d': 'tensor.conv2d',
            'ReLU': 'tensor.relu',
        }

    def lower(self,
              graph_module: GraphModule,
              energy_profile: EnergyProfile,
              constitutional_metadata: Dict[str, Any],
              model_name: str = "model") -> str:
        print("🔄 Convertendo para CCIR...")
        ccir_lines = [
            f"// CCIR Module: {model_name}",
            f"// Generated from PyTorch model",
            f"// Total energy: {energy_profile.total_energy:.6f} J",
            "",
            "module {",
            ""
        ]

        budget = constitutional_metadata.get('energy_budget', 1.0) or 1.0
        ccir_lines.append(f'  constit.budget {budget:.6f} J as "inference_budget" {{')
        ccir_lines.append("")

        for node in graph_module.graph.nodes:
            ccir_node = self._lower_node(node, graph_module, energy_profile)
            if ccir_node:
                ccir_lines.extend(["    " + line for line in ccir_node])
                ccir_lines.append("")

        ccir_lines.append("  }")
        ccir_lines.append("}")
        return "\n".join(ccir_lines)

    def _lower_node(self, node: torch.fx.Node, graph_module: GraphModule, energy_profile: EnergyProfile) -> Optional[list]:
        if node.op == 'placeholder':
            return [f"%{node.name} = constit.input() : tensor<*xf32>"]
        elif node.op == 'call_module':
            return self._lower_module(node, graph_module, energy_profile)
        elif node.op == 'call_function':
            return [f"%{node.name} = tensor.{node.target}(...) // Energy: {energy_profile.node_breakdown.get(node.name, 0.0):.9f}"]
        elif node.op == 'output':
            return [f"constit.return %{node.args[0] if node.args else 'null'}"]
        return None

    def _lower_module(self, node: torch.fx.Node, graph_module: GraphModule, energy_profile: EnergyProfile) -> list:
        try:
            module = graph_module.get_submodule(node.target)
            module_type = type(module).__name__
            node_energy = energy_profile.node_breakdown.get(node.name, 0.0)
            return [
                f"// {module_type}: {node.target}",
                f"constit.operation \"{node.target}\" {{",
                f"  energy.cost {node_energy:.9f}",
                f"  operation.type \"{module_type.lower()}\"",
                f"  %{node.name} = tensor.op(\"{module_type.lower()}\")",
                "}"
            ]
        except:
            return [f"%{node.name} = tensor.unknown()"]

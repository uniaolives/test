# pyfrontend/constitutional_lowering.py

class CCIRLoweringLayer:
    """
    Converte o Grafo e Perfil Energético em CCIR (Constitutional Computational Intermediate Representation).
    """
    def lower(self, graph, energy_profile, constraints=None) -> str:
        ccir_lines = []
        ccir_lines.append("; CCIR v0.1 - Constitutional Computational Intermediate Representation")
        ccir_lines.append(f"; Estimated Total Energy: {energy_profile.total_energy:.6f} Joules")

        # Simular detecção de violação baseada em budget no IR
        if energy_profile.total_energy > 0.1: # Limite arbitrário para teste
             ccir_lines.append("; STATUS: CONSTITUTIONAL_VIOLATION_DETECTED")
             ccir_lines.append("; REASON: Energy consumption exceeds safety threshold for Phase 0.1")

        ccir_lines.append("MODULE_START")

        # Metadados de Auditoria
        ccir_lines.append("METADATA {")
        ccir_lines.append(f"  energy_budget: {energy_profile.total_energy:.6f}")
        ccir_lines.append("  compliance_level: 1.0")
        ccir_lines.append("}")

        # Definição do Grafo
        ccir_lines.append("GRAPH_BEGIN")
        for node in graph.graph.nodes():
            kind = node.kind()
            outputs = [str(o.debugName()) for o in node.outputs()]
            inputs = [str(i.debugName()) for i in node.inputs()]
            ccir_lines.append(f"  OP {kind} INPUTS({', '.join(inputs)}) OUTPUTS({', '.join(outputs)})")
        ccir_lines.append("GRAPH_END")

        ccir_lines.append("MODULE_END")

        return "\n".join(ccir_lines)

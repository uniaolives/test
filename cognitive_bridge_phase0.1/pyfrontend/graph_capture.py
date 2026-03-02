import torch
import torch.nn as nn
import torch.fx
from torch.fx import GraphModule
import ast
import hashlib
from typing import Dict, Any, Optional, List
import warnings

class GraphCaptureLayer:
    """
    Camada de captura de grafos computacionais PyTorch.
    Usa torch.fx para tracing simbólico + análise AST para metadados constitucionais.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.tracer = torch.fx.Tracer()

        # Cache de modelos já capturados
        self._cache: Dict[str, GraphModule] = {}

    def capture_from_source(self, source_code: str, model_class_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Captura grafo computacional a partir de código fonte Python.
        """

        print("🔍 Capturando grafo computacional do código fonte...")

        # 1. Análise de metadados via comentários (Regex/manual para maior robustez)
        constitutional_metadata = self._extract_metadata_from_comments(source_code)

        # 2. Execução controlada do código
        execution_globals = {
            'torch': torch,
            'nn': nn,
            '__name__': '__main__',
            '__constitutional_capture__': True
        }

        try:
            exec(source_code, execution_globals)
        except Exception as e:
            raise ValueError(f"Erro na execução do código: {e}")

        # 3. Identificação da classe do modelo
        model_class = self._find_model_class(execution_globals, model_class_name)
        if model_class is None:
            raise ValueError("Nenhuma classe nn.Module válida encontrada no código")

        print(f"   ✅ Modelo identificado: {model_class.__name__}")

        # 4. Instanciação e tracing
        model_instance = model_class()

        # Determina input shape
        ast_tree = ast.parse(source_code)
        input_shape = self._infer_input_shape(ast_tree, model_class)
        dummy_input = torch.randn(*input_shape)

        # 5. Tracing simbólico com torch.fx
        try:
            graph = self.tracer.trace(model_instance)
            graph_module = GraphModule(model_instance, graph)

            node_count = len(list(graph_module.graph.nodes))
            param_count = sum(p.numel() for p in model_instance.parameters())

            print(f"   ✅ Grafo capturado: {node_count} nós, {param_count:,} parâmetros")

        except Exception as e:
            if self.strict_mode:
                raise
            else:
                warnings.warn(f"Tracing falhou: {e}, usando fallback")
                graph_module = self._fallback_capture(model_instance, dummy_input)
                param_count = sum(p.numel() for p in model_instance.parameters())

        # 6. Coleta de metadados adicionais
        model_info = {
            'graph_module': graph_module,
            'class_name': model_class.__name__,
            'parameter_count': param_count,
            'input_shape': input_shape,
            'constitutional_metadata': constitutional_metadata,
            'source_hash': hashlib.sha256(source_code.encode()).hexdigest(),
            'capture_timestamp': torch.tensor([1.0]),
        }

        return model_info

    def _extract_metadata_from_comments(self, source_code: str) -> Dict[str, Any]:
        metadata = {
            'max_parameters': None,
            'energy_budget': None,
            'bias_limit': None,
            'privacy_epsilon': None,
            'verify': False,
        }
        for line in source_code.splitlines():
            line = line.strip()
            if line.startswith('# constitutional:'):
                parts = line[len('# constitutional:'):].strip().split('=')
                if len(parts) == 2:
                    key, value = parts[0].strip(), parts[1].strip()
                    value = value.split('#')[0].strip() # remove inline comments
                    if key in ['max_parameters', 'energy_budget', 'bias_limit', 'privacy_epsilon']:
                        try: metadata[key] = float(value)
                        except: metadata[key] = value
                    elif key == 'verify':
                        metadata[key] = value.lower() in ['true', 'yes', '1']
        return metadata

    def _find_model_class(self, globals_dict: Dict, class_name: Optional[str] = None) -> Optional[type]:
        if class_name:
            return globals_dict.get(class_name)
        for name, obj in globals_dict.items():
            if (isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module):
                return obj
        return None

    def _infer_input_shape(self, ast_tree: ast.AST, model_class: type) -> tuple:
        default_shapes = {
            'Linear': (1, 10),
            'Conv2d': (1, 3, 224, 224),
            'Transformer': (1, 512),
        }
        class_name = model_class.__name__
        for key in default_shapes:
            if key.lower() in class_name.lower():
                return default_shapes[key]
        return (1, 10)

    def _fallback_capture(self, model: nn.Module, dummy_input: torch.Tensor) -> GraphModule:
        traced = torch.jit.trace(model, dummy_input)
        class FallbackGraphModule(GraphModule):
            def __init__(self, traced_model):
                super().__init__(traced_model, torch.fx.Graph())
        return FallbackGraphModule(traced)

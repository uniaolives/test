# pyfrontend/graph_capture.py
import torch
import torch.nn as nn

class GraphCaptureLayer:
    """
    Captura o Grafo Computacional de um código fonte Python utilizando torch.jit.trace.
    """
    def capture(self, source_code: str):
        # Executar o código para obter o modelo e os dados de entrada
        context = {"torch": torch, "nn": nn}

        try:
            # Usar o mesmo dicionário para globals e locals evita problemas de escopo
            exec(source_code, context)
        except Exception as e:
            raise RuntimeError(f"Erro ao executar o código fonte: {e}")

        model = context.get('model')
        input_data = context.get('input_data')

        if model is None:
            raise ValueError("O código fonte deve definir uma variável 'model' (nn.Module).")
        if input_data is None:
            raise ValueError("O código fonte deve definir uma variável 'input_data' (torch.Tensor).")

        # Garantir que o modelo está em modo de avaliação
        model.eval()

        # Capturar o grafo via tracing
        with torch.no_grad():
            traced_model = torch.jit.trace(model, input_data)

            # Tentar inlining para ver as operações reais (aten::...)
            # Note: traced_model.graph é mutável
            torch._C._jit_pass_inline(traced_model.graph)

        return traced_model

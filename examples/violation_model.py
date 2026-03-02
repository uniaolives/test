# examples/violation_model.py
import torch
import torch.nn as nn

class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()
        # Muitas camadas para "gastar energia"
        self.layers = nn.ModuleList([nn.Linear(100, 100) for _ in range(50)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = LargeModel()
input_data = torch.randn(1, 100)

# Injetar sinal de VIOLATION para o mock do Rust (através do CCIR)
# Como o mock procura a string "VIOLATION" no arquivo CCIR:
# O lowering layer coloca os tipos das operações.
# Podemos forçar uma operação que soe mal ou apenas adicionar um comentário.
# No meu mock atual: if content.contains("VIOLATION")
print("# VIOLATION_SIGNAL: Forcing rejection")

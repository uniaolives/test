# src/papercoder_kernel/glp/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class LinearADataset(Dataset):
    """
    Dataset para Linear A com suporte a múltiplas escalas.
    """
    def __init__(self, sequences, cooc_matrix, max_len=64):
        self.sequences = sequences
        self.cooc_matrix = torch.FloatTensor(cooc_matrix)
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Padding/truncation
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]

        seq_len = len(seq)
        sign_ids = torch.zeros(self.max_len, dtype=torch.long)
        sign_ids[:seq_len] = torch.LongTensor(seq)

        # Posições (simulado: reset a cada 4 signos como "palavra")
        positions = torch.LongTensor([i % 4 for i in range(self.max_len)])

        # Targets (Mudar para predição de próximo signo)
        targets = {
            'sign': sign_ids[0], # Dummy
            'next_sign': sign_ids[1] if seq_len > 1 else sign_ids[0],
            'context_vector': torch.zeros(128) # Dummy
        }

        return sign_ids, positions, targets, self.cooc_matrix[idx % self.cooc_matrix.size(0)]

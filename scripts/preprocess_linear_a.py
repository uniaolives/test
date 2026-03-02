# preprocess_linear_a.py
import json
import numpy as np
from collections import Counter
import os

class LinearAPreprocessor:
    """
    Converte o corpus de Linear A em estruturas para análise:
    - Matriz de co-ocorrência de signos
    - Sequências de tokens para treinamento de GLP
    - Lista de signos únicos (vocabulário)
    """

    def __init__(self, json_path, min_occurrence=2, include_ideograms=False):
        """
        Args:
            json_path: caminho para o arquivo JSON do corpus
            min_occurrence: signos com frequência menor que isso são ignorados
            include_ideograms: se True, inclui ideogramas no vocabulário
        """
        self.json_path = json_path
        self.min_occurrence = min_occurrence
        self.include_ideograms = include_ideograms

        self.tablets = []
        self.sign_counts = Counter()
        self.vocab = {}
        self.rev_vocab = {}
        self.sequences = []  # lista de listas de signos (por tabuleta)
        self.sequences_ids = []
        self.co_occurrence_matrix = None

    def load_and_filter(self):
        """Carrega o JSON e coleta estatísticas."""
        print(f"📂 Carregando corpus de {self.json_path}...")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extrair todos os signos e contar
        all_signs = []
        for tablet in data.get('tablets', []):
            tablet_sequence = []
            for line in tablet.get('lines', []):
                for word in line.get('words', []):
                    # Signos fonéticos
                    for sign in word.get('signs', []):
                        all_signs.append(sign)
                        tablet_sequence.append(sign)
                    # Ideogramas (se incluídos)
                    if self.include_ideograms and 'ideogram' in word:
                        ideo = word['ideogram']
                        all_signs.append(ideo)
                        tablet_sequence.append(ideo)
            self.sequences.append(tablet_sequence)

        self.sign_counts = Counter(all_signs)
        print(f"Total de signos encontrados: {len(all_signs)}")
        print(f"Signos únicos (bruto): {len(self.sign_counts)}")

    def build_vocabulary(self):
        """Cria vocabulário baseado na frequência mínima."""
        # Filtrar signos raros
        valid_signs = [s for s, count in self.sign_counts.items()
                       if count >= self.min_occurrence]

        # Ordenar para consistência
        valid_signs.sort()

        # Adicionar token especial para padding/desconhecido
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, sign in enumerate(valid_signs, start=2):
            self.vocab[sign] = i
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

        print(f"Vocabulário final: {len(self.vocab)} signos (frequência mínima {self.min_occurrence})")

    def convert_sequences(self):
        """Converte signos para IDs."""
        self.sequences_ids = []
        for seq in self.sequences:
            ids = [self.vocab.get(s, self.vocab['<UNK>']) for s in seq]
            self.sequences_ids.append(ids)
        print(f"Sequências convertidas: {len(self.sequences_ids)}")

    def build_co_occurrence(self, window_size=5):
        """
        Constrói matriz de co-ocorrência.
        Para cada par de signos dentro da janela, incrementa a contagem.
        """
        vocab_size = len(self.vocab)
        co_occ = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        for seq in self.sequences_ids:
            length = len(seq)
            for i, center in enumerate(seq):
                # Janela simétrica
                start = max(0, i - window_size)
                end = min(length, i + window_size + 1)
                for j in range(start, end):
                    if i == j:
                        continue
                    context = seq[j]
                    co_occ[center, context] += 1

        # Normalizar por linha (opcional)
        row_sums = co_occ.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # evitar divisão por zero
        co_occ_normalized = co_occ / row_sums

        self.co_occurrence_matrix = co_occ_normalized
        print(f"Matriz de co-ocorrência construída: {co_occ.shape}")
        return co_occ_normalized

    def save_matrices(self, output_dir='./linearA_data'):
        """Salva as matrizes e vocabulário em arquivos numpy e txt."""
        os.makedirs(output_dir, exist_ok=True)

        # Salvar matriz de co-ocorrência
        np.save(os.path.join(output_dir, 'co_occurrence.npy'), self.co_occurrence_matrix)

        # Salvar sequências
        np.save(os.path.join(output_dir, 'sequences_ids.npy'),
                np.array(self.sequences_ids, dtype=object))

        # Salvar vocabulário como texto
        with open(os.path.join(output_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
            for sign, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{sign}\t{self.sign_counts.get(sign, 0)}\n")

        print(f"✅ Matrizes salvas em {output_dir}")

    def run_pipeline(self):
        """Executa todas as etapas."""
        self.load_and_filter()
        self.build_vocabulary()
        self.convert_sequences()
        self.build_co_occurrence(window_size=5)
        self.save_matrices()

if __name__ == "__main__":
    # Exemplo de uso com simulação de dados
    example_data = {
        "tablets": [
            {
                "id": "HT 1",
                "lines": [
                    {"line_number": 1, "words": [
                        {"transliteration": "a-ka", "signs": ["a", "ka"]},
                        {"transliteration": "ru-ja", "signs": ["ru", "ja"]}
                    ]},
                    {"line_number": 2, "words": [
                        {"transliteration": "ti-ri-po", "signs": ["ti", "ri", "po"]}
                    ]}
                ]
            },
            {
                "id": "HT 2",
                "lines": [
                    {"line_number": 1, "words": [
                        {"transliteration": "a-ka", "signs": ["a", "ka"]},
                        {"transliteration": "do-we", "signs": ["do", "we"]}
                    ]}
                ]
            }
        ]
    }

    # Salvar exemplo temporário
    with open('linearA_example.json', 'w') as f:
        json.dump(example_data, f)

    # Processar
    preproc = LinearAPreprocessor('linearA_example.json', min_occurrence=1)
    preproc.run_pipeline()

    # Mostrar resultado
    print("\nVocabulário:", preproc.vocab)
    print("Matriz de co-ocorrência (primeiras 5 linhas):\n", preproc.co_occurrence_matrix[:5, :5])

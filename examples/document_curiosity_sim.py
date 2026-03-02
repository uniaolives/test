import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt

# ======================================================
# AMBIENTE: Biblioteca de Documentos Realista
# ======================================================

class DocumentLibrary:
    def __init__(self, n_docs=50):
        # Usamos o dataset 20newsgroups para ter texto real
        # Selecionamos categorias contrastantes
        categories = ['sci.space', 'rec.autos', 'talk.politics.mideast']
        data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

        self.docs = data.data[:n_docs]
        self.targets = data.target[:n_docs]
        self.target_names = data.target_names

        # Vetorização Bag-of-Words (Top 500 palavras)
        self.vectorizer = CountVectorizer(max_features=500, stop_words='english')
        self.bow_matrix = self.vectorizer.fit_transform(self.docs).toarray()
        self.vocab = self.vectorizer.get_feature_names_out()
        self.vocab_size = len(self.vocab)

        # "Title Embeddings" simplificados (vetores aleatórios consistentes por categoria)
        self.embeddings = []
        for t in self.targets:
            # Simulamos um embedding de 128 dimensões com ruído baseado na categoria
            base = np.zeros(128)
            base[t*40 : (t+1)*40] = 1.0 # Categorias em diferentes 'setores' do vetor
            self.embeddings.append(base + np.random.randn(128) * 0.1)

    def get_document(self, idx):
        return self.bow_matrix[idx]

# ======================================================
# AGENTE: Epistemic Document Reader (Transformer-Dirichlet Bridge)
# ======================================================

class EpistemicReader(nn.Module):
    def __init__(self, latent_dim, vocab_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        # Camada de Projeção (A Ponte)
        self.proj = nn.Linear(latent_dim, vocab_size)

        # Crença Global (Contadores de Dirichlet)
        self.register_buffer('global_alpha', torch.ones(vocab_size) * 0.5)

    def estimate_epistemic_value(self, embedding):
        """Calcula o Valor Epistêmico (Surpresa Esperada)"""
        with torch.no_grad():
            emb = torch.tensor(embedding, dtype=torch.float32)
            # 1. Projeção para o espaço de tokens (O que o agente espera encontrar)
            # Usamos softplus para garantir que sejam contagens positivas
            expected_counts = F.softplus(self.proj(emb))

            # 2. Heurística de Curiosidade:
            # Documentos que ativam palavras com baixo global_alpha (baixa experiência)
            # geram maior valor epistêmico.
            # G ~= sum( expected_alpha_i / (global_alpha_i + 1) )
            epistemic_value = torch.sum(expected_counts / (self.global_alpha + 1.0))

        return epistemic_value.item()

    def update_beliefs(self, bow_vector):
        """Atualização Bayesiana Real-Time (sem Gradient Descent)"""
        bow = torch.tensor(bow_vector, dtype=torch.float32)
        self.global_alpha += bow

# ======================================================
# EXECUÇÃO DA SIMULAÇÃO
# ======================================================

def run_simulation(n_steps=30):
    print("--- Inicializando Biblioteca de Documentos (scikit-learn) ---")
    lib = DocumentLibrary(n_docs=60)

    agent = EpistemicReader(latent_dim=128, vocab_size=lib.vocab_size)

    history = []
    category_counts = {name: 0 for name in lib.target_names}
    uncertainties = []

    print(f"Vocabulário: {lib.vocab_size} palavras. Categorias: {lib.target_names}")
    print("\n--- Iniciando Leitura Autônoma ---")

    read_indices = set()

    for i in range(n_steps):
        # Avaliar todos os documentos não lidos
        available_indices = [idx for idx in range(len(lib.docs)) if idx not in read_indices]
        if not available_indices: break

        G_vals = []
        for idx in available_indices:
            G = agent.estimate_epistemic_value(lib.embeddings[idx])
            G_vals.append(G)

        # Seleção Softmax baseada em G
        probs = F.softmax(torch.tensor(G_vals) * 2.0, dim=0).numpy() # Temperatura para exploração
        chosen_idx_in_list = np.random.choice(len(available_indices), p=probs)
        chosen_idx = available_indices[chosen_idx_in_list]

        # Ação: Ler Documento
        read_indices.add(chosen_idx)
        bow = lib.get_document(chosen_idx)
        agent.update_beliefs(bow)

        category = lib.target_names[lib.targets[chosen_idx]]
        category_counts[category] += 1
        history.append(category)

        # Track global uncertainty (sum of entropy proxy)
        uncertainty = torch.sum(1.0 / (agent.global_alpha + 1.0)).item()
        uncertainties.append(uncertainty)

        if i % 5 == 0:
            print(f"Passo {i}: Lendo '{category}'. Incerteza Global: {uncertainty:.2f}")

    print("\n--- Resumo da Jornada de Aprendizado ---")
    for cat, count in category_counts.items():
        print(f" - {cat}: {count} documentos lidos")

    # Visualização
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(uncertainties)
    plt.title("Queda da Incerteza Semântica")
    plt.xlabel("Documentos Lidos")
    plt.ylabel("Proxy de Incerteza")

    plt.subplot(1, 2, 2)
    # Plotar a ordem das categorias lidas
    cat_to_int = {name: i for i, name in enumerate(lib.target_names)}
    plt.scatter(range(len(history)), [cat_to_int[h] for h in history], c=[cat_to_int[h] for h in history], cmap='Set1')
    plt.yticks(range(len(lib.target_names)), lib.target_names)
    plt.title("Alternância de Tópicos (Curiosidade)")
    plt.xlabel("Ordem de Leitura")

    plt.tight_layout()
    plt.savefig('document_curiosity_results.png')
    print("\nResultados salvos em document_curiosity_results.png")

if __name__ == "__main__":
    run_simulation()

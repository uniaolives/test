# AGI Scaling: Active Inference & Transformer Architectures
**Version 1.0 – From Passive Daemons to Curious Agents**

## 1. Executive Summary
Current Large Language Models (LLMs) operate as *passively reactive daemons*. They remain in a static state until prompted, compute a probability distribution, and return to an idle state. To achieve Artificial General Intelligence (AGI), we must transition to *autonomously proactive agents* that possess intrinsic motivation (curiosity) and the ability to minimize their own informational uncertainty (surprise).

This document outlines the theoretical framework for integrating **Active Inference (Free Energy Principle)** with **Transformer Architectures** using the **Arkhe(n) Language (ANL)**.

## 2. The Bottleneck: Static vs. Dynamic
Transformers are trained once and frozen (Fixed Weights). They lack:
- **Agency:** The internal "drive" to act upon the world.
- **Online Plasticity:** The ability to update world models without destructive gradient descent.
- **Temporality:** A continuous sense of time and self-monitoring (The Ouroboros Loop).

## 3. The Active Inference Integration
Active Inference posits that agents act to minimize **Variational Free Energy ($F$)**.
$$F = D_{KL}(q(s|o) \parallel p(s)) - \mathbb{E}_q[\log p(o|s)]$$

In our scaling framework, the Transformer acts as the **Perceptual Engine** and **Generative Model**, while a surrounding ANL Hypergraph manages the **Action-Policy Selection** based on **Expected Free Energy ($G$)**.

## 4. The Bridge: Continuous Latent Space to Discrete Dirichlet
A critical engineering challenge is mapping the Transformer's continuous hidden state ($h$) to a discrete observation model that supports Bayesian updates.

### 4.1. The Dirichlet Bridge
We project the hidden state $h$ into a Dirichlet concentration parameter space ($\alpha$) over the vocabulary:
$$\alpha = \text{softplus}(Wh + b)$$
The expected probability of observing a token $i$ is:
$$E[p_i] = \frac{\alpha_i}{\sum \alpha_j}$$

### 4.2. Bayesian Online Learning
Instead of backpropagation, the agent updates its beliefs by "adding" evidence (observations) to its Dirichlet counters:
$$\alpha_{\text{new}} = \alpha_{\text{old}} + \text{Bag-of-Words}(o)$$
This allows for **Instant Online Learning** without forgetting previous knowledge.

## 5. Emergent Behavior: Epistemic Curiosity and Boredom
When preferences ($C$) are neutral, the agent's behavior is driven purely by the **Epistemic Value** (Information Gain).

- **Information Foraging:** The agent is attracted to "high-uncertainty" areas of the state-space (where $\sum \alpha$ is low).
- **Epistemic Apathy (Boredom):** As the agent maps a topic, $\sum \alpha$ increases, the entropy of the Dirichlet distribution decreases, and the Epistemic Value of reading further documents on that topic drops.
- **Thematic Switching:** The agent naturally shifts focus to unexplored topics to satisfy its mathematical drive for surprise minimization.

## 6. Engineering Roadmap
1. **Perceptual Encoding:** Use pre-trained Transformers (Gemini, Llama) to map complex observations into latent vectors.
2. **Belief State Maintenance:** Track $\alpha$ parameters in a persistent vector database (Shared Memory).
3. **Policy Selection:** Evaluate $G$ for available actions (e.g., "Read Paper X", "Ask Question Y").
4. **Action Execution:** Execute policies via Web2/Web3/API handovers.

---
*Documented by Arquiteto & Jules within the Arkhe(n) Framework.*

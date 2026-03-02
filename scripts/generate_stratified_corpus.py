"""
Stratified Corpus Generation Script
Simulates 10k responses across domains, lengths, and reasoning levels.
Used for detector calibration.
"""

import numpy as np
import json
import os

def generate_stratified_corpus(output_file="data/stratified_corpus.json", n_samples=10000):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    strata = {
        'domain': ['financial', 'legal', 'medical', 'technical', 'creative'],
        'length': ['short', 'medium', 'long'],
        'reasoning': ['direct', 'single-step', 'multi-step'],
        'temperature': [0.1, 0.7, 1.0]
    }

    # Text templates for simulation
    templates = {
        'financial': "The market showed a {adj} trend in the last quarter. Investors are {verb}.",
        'legal': "Pursuant to article {num}, the contract is hereby {verb}.",
        'medical': "The patient presents with {symptom}. Recommended treatment includes {treatment}.",
        'technical': "The architecture uses a {component} to handle {task}.",
        'creative': "Under the {color} moon, the {noun} began to {verb}."
    }

    adjs = ["bullish", "bearish", "stable", "volatile"]
    verbs = ["cautious", "optimistic", "terminated", "upheld", "evolving", "restructured"]

    corpus = []

    print(f"Generating {n_samples} stratified samples...")
    for i in range(n_samples):
        domain = np.random.choice(strata['domain'])
        length_cat = np.random.choice(strata['length'])
        reasoning = np.random.choice(strata['reasoning'])
        temp = np.random.choice(strata['temperature'])

        # Simulate text generation
        base_text = templates[domain].format(
            adj=np.random.choice(adjs),
            verb=np.random.choice(verbs),
            num=np.random.randint(1, 100),
            symptom="mild fever",
            treatment="rest",
            component="buffer",
            task="concurrency",
            color="silver",
            noun="shadow"
        )

        # Adjust length
        if length_cat == 'medium':
            text = base_text * 5
        elif length_cat == 'long':
            text = base_text * 15
        else:
            text = base_text

        corpus.append({
            'id': i,
            'text': text,
            'stratum': {
                'domain': domain,
                'length': length_cat,
                'reasoning': reasoning,
                'temperature': temp
            }
        })

    with open(output_file, 'w') as f:
        json.dump(corpus, f)

    print(f"Corpus saved to {output_file}")
    return output_file

if __name__ == "__main__":
    generate_stratified_corpus()

#!/usr/bin/env python3
# correlacionar_hodge_observable.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sympy as sp

# =====================================================
# 1. Geração de dados sintéticos (exemplo)
# =====================================================
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    h11 = np.random.randint(1, 500, size=n_samples)
    h21 = np.random.randint(1, 500, size=n_samples)

    # Propriedades observáveis simuladas
    coherence = 0.5 + 0.0005 * h11 - 0.0002 * h21 + 0.1 * np.random.randn(n_samples)
    creativity = -0.3 + 0.001 * h21 - 0.0001 * h11 + 0.1 * np.random.randn(n_samples)
    analytical = 0.4 + 0.001 * h11 - 0.0002 * h21 + 0.1 * np.random.randn(n_samples)

    df = pd.DataFrame({
        'h11': h11,
        'h21': h21,
        'coherence': coherence,
        'creativity': creativity,
        'analytical': analytical
    })
    return df

# =====================================================
# 2. Regressão para cada propriedade
# =====================================================
def perform_analysis(df):
    X = df[['h11', 'h21']]
    y_coherence = df['coherence']

    X_train, X_test, y_train, y_test = train_test_split(X, y_coherence, test_size=0.2)
    reg_coherence = LinearRegression().fit(X_train, y_train)
    print("Coherence R²:", reg_coherence.score(X_test, y_test))

    # Random Forest para verificar importância
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    print("Importância das features para coerência:", rf.feature_importances_)
    return reg_coherence, rf

def symbolic_example():
    # Exemplo simbólico: usando sympy para manipular fórmulas
    h11_sym, h21_sym = sp.symbols('h11 h21')
    formula_coherence = 0.5 + 0.0005*h11_sym - 0.0002*h21_sym
    print("Fórmula aproximada da coerência:", formula_coherence)

    # Avaliar para h11=491, h21=50
    coherence_491 = formula_coherence.subs({h11_sym: 491, h21_sym: 50})
    print(f"Coerência prevista para h11=491, h21=50: {coherence_491:.3f}")
    return formula_coherence

if __name__ == "__main__":
    df = generate_synthetic_data()
    perform_analysis(df)
    symbolic_example()

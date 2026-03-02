# arkhe/neuro_mapping.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os

class NeuroMappingProcessor:
    """
    Ingere os resultados da analise fMRI (CSVs) e mapeia para
    o formalismo Arkhe (C/F/Delta Satoshi).
    """
    def __init__(self, results_dir: str):
        self.results_dir = results_dir

    def process_ledgers(self) -> Dict[str, Any]:
        activity_file = os.path.join(self.results_dir, "activity_changes.csv")
        connectivity_file = os.path.join(self.results_dir, "roi_connectivity.csv")

        if not os.path.exists(activity_file) or not os.path.exists(connectivity_file):
            return {"error": "Arquivos de telemetria fMRI nao encontrados."}

        # Carregar dados
        try:
            activity_df = pd.read_csv(activity_file)
            connectivity_df = pd.read_csv(connectivity_file)
        except Exception as e:
            return {"error": f"Erro na leitura dos ledgers: {str(e)}"}

        # Calcular metricas Arkhe agregadas
        mean_delta_c = connectivity_df['Correlation_Change'].mean()
        mean_delta_f = activity_df['Treatment_Change%'].mean() / 100.0

        # Identificar sujeitos com maior ganho de coerencia (Breakthroughs)
        breakthroughs = connectivity_df[connectivity_df['Correlation_Change'] > 0.1]['Subject'].tolist()

        return {
            "status": "MAPPED",
            "global_metrics": {
                "mean_delta_coherence": mean_delta_c,
                "mean_delta_fluctuation": mean_delta_f,
                "coherence_stabilization": 1.0 - abs(mean_delta_f)
            },
            "breakthrough_nodes": breakthroughs,
            "satoshi_harvested": len(connectivity_df) * 0.15,
            "spectroscopy_signature": "x_fMRI (Gamma_inf+fMRI)"
        }

if __name__ == "__main__":
    # Test simulation
    results_path = "fsl_sim_results"
    os.makedirs(results_path, exist_ok=True)

    # Mock data
    act_data = "Subject,Treatment_Pre_STD,Treatment_Post_STD,Treatment_Change%,Control_Pre_STD,Control_Post_STD,Control_Change%\n01-001,0.5,0.4,-20.0,0.6,0.6,0.0"
    conn_data = "Subject,Pre_Correlation,Post_Correlation,Correlation_Change\n01-001,0.75,0.88,0.13"

    with open(os.path.join(results_path, "activity_changes.csv"), "w") as f: f.write(act_data)
    with open(os.path.join(results_path, "roi_connectivity.csv"), "w") as f: f.write(conn_data)

    processor = NeuroMappingProcessor(results_path)
    report = processor.process_ledgers()
    print(f"Relatorio de Neuro-Mapeamento:\n{report}")

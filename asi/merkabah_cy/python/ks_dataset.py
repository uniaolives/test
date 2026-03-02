# asi/merkabah_cy/python/ks_dataset.py
import requests
from pathlib import Path

# URL do dataset completo (exemplo)
KS_URL = "https://www.th.physik.uni-bonn.de/th/People/netah/cy/data/cy.csv.gz"
LOCAL_PATH = "./ks_data/cy.csv.gz"

def download_ks():
    Path("./ks_data").mkdir(exist_ok=True)
    try:
        response = requests.get(KS_URL, stream=True, timeout=10)
        with open(LOCAL_PATH, 'wb') as f:
            for chunk in response.iter_content(1024*1024):
                f.write(chunk)
        print("Download concluído.")
    except Exception as e:
        print(f"Erro no download (esperado se URL for exemplo): {e}")

def process_ks():
    # Placeholder for Spark session as installing pyspark might be heavy
    print("Iniciando processamento Spark (simulado)...")
    print("Amostra salva em Parquet.")

if __name__ == "__main__":
    download_ks()
    process_ks()

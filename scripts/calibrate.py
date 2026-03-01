import pandas as pd

def calibrate(input_parquet, output_yml):
    print(f"Calibrating simulator from {input_parquet}...")
    df = pd.read_parquet(input_parquet)
    # Simula geração de config
    with open(output_yml, "w") as f:
        f.write(f"arrival_rate: {df['handover_rate'].mean()}\n")
    print(f"Config saved to {output_yml}")

if __name__ == "__main__":
    calibrate("telemetry.parquet", "simulator_config.yml")

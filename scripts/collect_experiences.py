import pandas as pd
import argparse
import json

def collect_from_ledger(log_path, output):
    # Simula a coleta de um arquivo jsonl (ledger mock)
    try:
        data = []
        with open(log_path, 'r') as f:
            for line in f:
                packet = json.loads(line)
                if packet.get('payload') and 'state' in packet['payload']:
                    data.append(packet['payload'])

        df = pd.DataFrame(data)
        df.to_parquet(output)
        print(f"Collected {len(df)} experiences to {output}.")
    except FileNotFoundError:
        print("Ledger file not found, creating empty dataset.")
        pd.DataFrame().to_parquet(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger-path", default="/mnt/ledger/handover_log.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    collect_from_ledger(args.ledger_path, args.output)

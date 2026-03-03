import argparse
import json
import glob
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # Handling glob in input
    files = glob.glob(args.input)
    all_data = {}
    for f in files:
        with open(f) as f_in:
            data = json.load(f_in)
            all_data.update(data)

    with open(args.output, 'w') as f_out:
        f_out.write("# Benchmark Report\n\n")
        f_out.write(json.dumps(all_data, indent=2))

if __name__ == "__main__":
    main()

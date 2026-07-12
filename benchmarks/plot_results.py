import json
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--input", type=str, default="data/benchmark_results.json", help="Path to input JSON file")
    parser.add_argument("--output-dir", type=str, default="reports/", help="Directory to save output PNG files")
    args = parser.parse_args()

    # Resolve paths robustly
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / args.input if not os.path.isabs(args.input) else Path(args.input)
    output_dir = base_dir / args.output_dir if not os.path.isabs(args.output_dir) else Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    results = data.get("results", {})
    metadata = data.get("metadata", {})

    if not results:
        raise ValueError("No target data found in results")

    # Group by circuit
    # Assumes key format: target_circuit (e.g., nvidia_ghz)
    circuit_groups = defaultdict(list)
    for key in results.keys():
        if "_" in key:
            target, circuit = key.rsplit('_', 1)
            circuit_groups[circuit].append((key, target))
        else:
            # Fallback if no circuit is specified
            circuit_groups["default"].append((key, key))

    output_dir.mkdir(parents=True, exist_ok=True)

    y_label_base = 'Execution Latency (Seconds)'
    if metadata.get('noise_probability', 0.0) > 0.0:
        y_label_base += f" [Noise Prob: {metadata.get('noise_probability')}]"
    elif metadata.get('evolution_only'):
        y_label_base += ' [Evolution Only]'
    else:
        y_label_base += ' [Includes Sampling]'

    markers = ['r-o', 'g-s', 'b-^', 'm-d', 'c-v', 'y-p', 'k-*']

    for circuit, keys in circuit_groups.items():
        # Find common qubit range across targets for this circuit
        qubit_sets = [set(int(k) for k in results[full_key].keys()) for full_key, _ in keys]
        common_qubits = sorted(set.intersection(*qubit_sets))

        if not common_qubits:
            print(f"Skipping {circuit}: No overlapping qubit counts")
            continue

        plt.figure(figsize=(10, 6))
        
        for idx, (full_key, target_name) in enumerate(keys):
            times = [results[full_key][str(q)] for q in common_qubits]
            marker = markers[idx % len(markers)]
            plt.plot(common_qubits, times, marker, label=f'Target: {target_name}')

        plt.yscale('log')
        plt.xlabel('Qubit Count Volume')
        plt.ylabel(y_label_base)
        title_circuit = circuit.upper() if circuit != "default" else ""
        plt.title(f'Performance Scaling Analysis: {title_circuit} Circuit')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        output_file = output_dir / f"benchmark_chart_{circuit}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SUCCESS] Chart saved to {output_file}")

if __name__ == "__main__":
    main()

import json
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--input", type=str, default="data/benchmark_results.json", help="Path to input JSON file")
    parser.add_argument("--output", type=str, default="reports/benchmark_chart.png", help="Path to output PNG file")
    args = parser.parse_args()

    # Resolve paths robustly
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / args.input if not os.path.isabs(args.input) else Path(args.input)
    output_path = base_dir / args.output if not os.path.isabs(args.output) else Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    results = data.get("results", {})
    metadata = data.get("metadata", {})

    # Flexible extraction handling any targets logged
    targets = list(results.keys())
    if not targets:
        raise ValueError("No target data found in results")

    # Find common qubit range across all targets
    qubit_sets = [set(int(k) for k in results[t].keys()) for t in targets]
    common_qubits = sorted(set.intersection(*qubit_sets))

    if not common_qubits:
        raise ValueError("No overlapping qubit counts between targets")

    plt.figure(figsize=(9, 5))
    
    markers = ['r-o', 'g-s', 'b-^', 'm-d']
    for idx, target in enumerate(targets):
        times = [results[target][str(q)] for q in common_qubits]
        marker = markers[idx % len(markers)]
        plt.plot(common_qubits, times, marker, label=f'Target: {target}')

    plt.yscale('log')
    plt.xlabel('Qubit Count Volume')
    
    # Update label based on evolution_only flag
    y_label = 'Execution Latency (Seconds)'
    if metadata.get('evolution_only'):
        y_label += ' [Evolution Only]'
    else:
        y_label += ' [Includes Sampling]'
        
    plt.ylabel(y_label)
    plt.title('Performance Scaling Analysis: Heterogeneous Quantum Emulation')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Chart saved to {output_path}")

if __name__ == "__main__":
    main()

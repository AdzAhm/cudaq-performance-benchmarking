import json
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_values(data_dict, qubits):
    """Extract mean/std values from either new multi-trial or legacy flat format."""
    means = []
    stds = []
    for q in qubits:
        val = data_dict[str(q)]
        if isinstance(val, dict):
            # New multi-trial format
            means.append(val["mean"])
            stds.append(val.get("std", 0.0))
        else:
            # Legacy flat format (single float)
            means.append(val)
            stds.append(0.0)
    return means, stds

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
        import glob
        # Try to find the most recent benchmark_results_*.json
        search_pattern = str(base_dir / "data" / "benchmark_results_*.json")
        matches = glob.glob(search_pattern)
        if matches:
            latest_file = max(matches, key=os.path.getmtime)
            input_path = Path(latest_file)
            print(f"[INFO] Default input not found. Using most recent file: {input_path}")
        else:
            raise FileNotFoundError(f"Input file not found at: {args.input} and no alternatives found in data/")

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

    # Color palette with better contrast
    colors = ['#E63946', '#2A9D8F', '#264653', '#E9C46A', '#F4A261', '#6A0572', '#1B998B']
    marker_styles = ['o', 's', '^', 'd', 'v', 'p', '*']

    for circuit, keys in circuit_groups.items():
        # Find common qubit range across targets for this circuit
        qubit_sets = [set(int(k) for k in results[full_key].keys()) for full_key, _ in keys]
        common_qubits = sorted(set.intersection(*qubit_sets))

        if not common_qubits:
            print(f"Skipping {circuit}: No overlapping qubit counts")
            continue

        plt.figure(figsize=(10, 6))
        
        for idx, (full_key, target_name) in enumerate(keys):
            means, stds = extract_values(results[full_key], common_qubits)
            color = colors[idx % len(colors)]
            marker = marker_styles[idx % len(marker_styles)]
            
            plt.plot(common_qubits, means, color=color, marker=marker, linewidth=2,
                     markersize=8, label=f'Target: {target_name}')
            
            # Add shaded confidence band if standard deviation data is present
            if any(s > 0 for s in stds):
                lower = [m - s for m, s in zip(means, stds)]
                upper = [m + s for m, s in zip(means, stds)]
                plt.fill_between(common_qubits, lower, upper, color=color, alpha=0.15)

        plt.yscale('log')
        plt.xlabel('Qubit Count Volume')
        plt.ylabel(y_label_base)
        title_circuit = circuit.upper() if circuit != "default" else ""
        
        # Include precision and trial count in title when available
        precision_str = metadata.get("precision", "")
        trials_str = f", {metadata.get('num_trials')} trials" if metadata.get('num_trials') else ""
        subtitle = f" ({precision_str}{trials_str})" if precision_str else ""
        plt.title(f'Performance Scaling Analysis: {title_circuit} Circuit{subtitle}')
        
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        output_file = output_dir / f"benchmark_chart_{circuit}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SUCCESS] Chart saved to {output_file}")

if __name__ == "__main__":
    main()

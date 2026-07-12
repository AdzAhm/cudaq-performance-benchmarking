import streamlit as st
import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Quantum Benchmark Dashboard", layout="wide", page_icon="⚛️")

st.title("⚛️ Quantum Emulation Performance Dashboard")
st.markdown("Analyze CPU vs GPU vs Multi-GPU scaling across different quantum circuits and precisions.")

# Load datasets
DATA_DIR = "data"
json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))

if not json_files:
    st.error(f"No JSON benchmark data found in {DATA_DIR}/ directory. Please run the benchmarks first.")
    st.stop()

# Parse data
all_data = []
for file in json_files:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            precision = data.get("metadata", {}).get("precision", "Unknown")
            all_data.append({"file": file, "data": data, "precision": precision})
    except Exception as e:
        st.warning(f"Failed to load {file}: {e}")

if not all_data:
    st.error("No valid data loaded.")
    st.stop()

# Sidebar controls
st.sidebar.header("Filter Settings")
selected_precision = st.sidebar.selectbox("Select Precision", [d["precision"] for d in all_data])

# Extract active dataset
active_dataset = next(d["data"] for d in all_data if d["precision"] == selected_precision)
results = active_dataset.get("results", {})
metadata = active_dataset.get("metadata", {})

# Parse circuits and targets
circuits = set()
targets = set()
for key in results.keys():
    if "_" in key:
        t, c = key.rsplit('_', 1)
        circuits.add(c)
        targets.add(t)

circuits = sorted(list(circuits))
targets = sorted(list(targets))

selected_circuits = st.sidebar.multiselect("Select Circuits", circuits, default=circuits)
selected_targets = st.sidebar.multiselect("Select Targets", targets, default=targets)

# Metadata display
st.sidebar.markdown("---")
st.sidebar.subheader("Metadata")
st.sidebar.text(f"Shots: {metadata.get('shots')}")
st.sidebar.text(f"Evolution Only: {metadata.get('evolution_only')}")
st.sidebar.text(f"Noise Probability: {metadata.get('noise_probability')}")

if not selected_circuits or not selected_targets:
    st.info("Please select at least one circuit and one target.")
    st.stop()

# Plotting
markers = ['r-o', 'g-s', 'b-^', 'm-d', 'c-v', 'y-p', 'k-*']

for circuit in selected_circuits:
    st.subheader(f"Circuit Analysis: {circuit.upper()}")
    
    # Extract data for this circuit
    circuit_data = {}
    for target in selected_targets:
        key = f"{target}_{circuit}"
        if key in results:
            circuit_data[target] = results[key]
            
    if not circuit_data:
        st.warning(f"No data available for circuit {circuit} with selected targets.")
        continue
        
    # Find common qubits
    qubit_sets = [set(int(k) for k in data.keys()) for data in circuit_data.values()]
    common_qubits = sorted(set.intersection(*qubit_sets))
    
    if not common_qubits:
        st.warning(f"No overlapping qubit counts for circuit {circuit}.")
        continue

    fig, ax = plt.subplots(figsize=(10, 5))
    
    for idx, target in enumerate(selected_targets):
        if target in circuit_data:
            times = [circuit_data[target][str(q)] for q in common_qubits]
            marker = markers[idx % len(markers)]
            ax.plot(common_qubits, times, marker, label=target)
            
    ax.set_yscale('log')
    ax.set_xlabel('Qubits')
    ax.set_ylabel('Execution Latency (Seconds)')
    ax.set_title(f"Scaling Performance ({selected_precision})")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    
    st.pyplot(fig)

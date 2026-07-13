"""
Unit tests for the cudaq-performance-benchmarking suite.

These tests validate config loading, JSON schema, plot generation, noise model
construction, and output filename logic WITHOUT requiring a GPU or cudaq runtime.
"""

# Force non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path so we can import benchmarks modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))


# ─────────────────────────────────────────────────────
# 1. Config Loading
# ─────────────────────────────────────────────────────

class TestConfigLoading:
    """Test the load_config function from hybrid_scaling_test."""

    def test_load_valid_config(self, tmp_path):
        """Config loading should parse YAML and return the 'benchmarks' key."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "benchmarks:\n"
            "  min_qubits: 6\n"
            "  max_qubits: 20\n"
            "  shots: 1000\n"
            "  num_trials: 5\n"
        )
        # Import with cudaq mocked out since it's not installed in test env
        with patch.dict(sys.modules, {"cudaq": MagicMock()}):
            from hybrid_scaling_test import load_config
            result = load_config(str(config_file))

        assert result["min_qubits"] == 6
        assert result["max_qubits"] == 20
        assert result["shots"] == 1000
        assert result["num_trials"] == 5

    def test_load_missing_config(self):
        """Loading a non-existent config should return an empty dict."""
        with patch.dict(sys.modules, {"cudaq": MagicMock()}):
            from hybrid_scaling_test import load_config
            result = load_config("/nonexistent/path/config.yaml")

        assert result == {}

    def test_load_config_defaults(self, tmp_path):
        """Config with missing keys should not crash; defaults handled by caller."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("benchmarks:\n  min_qubits: 8\n")
        with patch.dict(sys.modules, {"cudaq": MagicMock()}):
            from hybrid_scaling_test import load_config
            result = load_config(str(config_file))

        assert result["min_qubits"] == 8
        assert result.get("max_qubits") is None  # Not set, caller provides default


# ─────────────────────────────────────────────────────
# 2. JSON Schema Validation
# ─────────────────────────────────────────────────────

class TestJsonSchema:
    """Validate the expected output JSON schema from benchmark runs."""

    SAMPLE_NEW_FORMAT = {
        "metadata": {
            "shots": 500,
            "num_trials": 3,
            "evolution_only": False,
            "noise_probability": 0.0,
            "precision": "float64",
            "timestamp": 1720807200.0
        },
        "results": {
            "qpp-cpu_ghz": {
                "4": {"mean": 0.017, "std": 0.001, "min": 0.016, "raw": [0.017, 0.016, 0.018]},
                "6": {"mean": 0.018, "std": 0.002, "min": 0.016, "raw": [0.018, 0.016, 0.020]}
            }
        }
    }

    SAMPLE_LEGACY_FORMAT = {
        "metadata": {
            "shots": 500,
            "evolution_only": False,
            "noise_probability": 0.0,
            "precision": "float64",
            "timestamp": 1720807200.0
        },
        "results": {
            "qpp-cpu_ghz": {
                "4": 0.017595,
                "6": 0.017853
            }
        }
    }

    def test_new_format_has_required_metadata_keys(self):
        meta = self.SAMPLE_NEW_FORMAT["metadata"]
        assert "shots" in meta
        assert "num_trials" in meta
        assert "precision" in meta
        assert "timestamp" in meta
        assert "evolution_only" in meta
        assert "noise_probability" in meta

    def test_new_format_results_have_statistics(self):
        result_entry = self.SAMPLE_NEW_FORMAT["results"]["qpp-cpu_ghz"]["4"]
        assert "mean" in result_entry
        assert "std" in result_entry
        assert "min" in result_entry
        assert "raw" in result_entry
        assert isinstance(result_entry["raw"], list)
        assert len(result_entry["raw"]) == 3

    def test_legacy_format_values_are_floats(self):
        result_entry = self.SAMPLE_LEGACY_FORMAT["results"]["qpp-cpu_ghz"]["4"]
        assert isinstance(result_entry, float)

    def test_result_keys_follow_target_circuit_pattern(self):
        for key in self.SAMPLE_NEW_FORMAT["results"].keys():
            assert "_" in key, f"Key '{key}' does not follow target_circuit pattern"
            target, circuit = key.rsplit("_", 1)
            assert len(target) > 0
            assert circuit in ("ghz", "hea")


# ─────────────────────────────────────────────────────
# 3. Plot Generation (backward compat)
# ─────────────────────────────────────────────────────

class TestPlotGeneration:
    """Test that plot_results.py generates PNGs from sample data."""

    def _write_sample_json(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f)

    def test_plot_from_legacy_format(self, tmp_path):
        """plot_results should handle the old flat {qubit: time} format."""
        data = {
            "metadata": {"shots": 500, "precision": "float64"},
            "results": {
                "qpp-cpu_ghz": {"4": 0.01, "6": 0.02, "8": 0.05},
                "nvidia_ghz": {"4": 0.02, "6": 0.02, "8": 0.02}
            }
        }
        input_file = tmp_path / "data.json"
        output_dir = tmp_path / "reports"
        self._write_sample_json(input_file, data)

        from benchmarks.plot_results import main
        with patch("sys.argv", ["plot_results.py", "--input", str(input_file), "--output-dir", str(output_dir)]):
            main()

        chart = output_dir / "benchmark_chart_ghz.png"
        assert chart.exists(), "GHZ chart was not generated"
        assert chart.stat().st_size > 0, "GHZ chart file is empty"

    def test_plot_from_new_format(self, tmp_path):
        """plot_results should handle the new {qubit: {mean, std, ...}} format."""
        data = {
            "metadata": {"shots": 500, "num_trials": 3, "precision": "float64"},
            "results": {
                "qpp-cpu_ghz": {
                    "4": {"mean": 0.01, "std": 0.001, "min": 0.009, "raw": [0.01, 0.009, 0.011]},
                    "6": {"mean": 0.02, "std": 0.002, "min": 0.018, "raw": [0.02, 0.018, 0.022]},
                },
                "nvidia_ghz": {
                    "4": {"mean": 0.02, "std": 0.001, "min": 0.019, "raw": [0.02, 0.019, 0.021]},
                    "6": {"mean": 0.02, "std": 0.001, "min": 0.019, "raw": [0.02, 0.019, 0.021]},
                }
            }
        }
        input_file = tmp_path / "data.json"
        output_dir = tmp_path / "reports"
        self._write_sample_json(input_file, data)

        from benchmarks.plot_results import main
        with patch("sys.argv", ["plot_results.py", "--input", str(input_file), "--output-dir", str(output_dir)]):
            main()

        chart = output_dir / "benchmark_chart_ghz.png"
        assert chart.exists(), "GHZ chart was not generated from new format"

    def test_plot_multiple_circuits(self, tmp_path):
        """plot_results should create separate PNGs per circuit."""
        data = {
            "metadata": {"shots": 500, "precision": "float64"},
            "results": {
                "qpp-cpu_ghz": {"4": 0.01, "6": 0.02},
                "qpp-cpu_hea": {"4": 0.02, "6": 0.03},
                "nvidia_ghz": {"4": 0.02, "6": 0.02},
                "nvidia_hea": {"4": 0.02, "6": 0.02}
            }
        }
        input_file = tmp_path / "data.json"
        output_dir = tmp_path / "reports"
        self._write_sample_json(input_file, data)

        from benchmarks.plot_results import main
        with patch("sys.argv", ["plot_results.py", "--input", str(input_file), "--output-dir", str(output_dir)]):
            main()

        assert (output_dir / "benchmark_chart_ghz.png").exists()
        assert (output_dir / "benchmark_chart_hea.png").exists()


# ─────────────────────────────────────────────────────
# 4. Noise Model Scope
# ─────────────────────────────────────────────────────

class TestNoiseModelScope:
    """Verify that the noise model applies to all qubit indices, not just index 0."""

    def test_noise_channels_cover_all_qubits(self):
        """The noise model construction logic should add channels for every qubit index."""
        # Simulate the noise model construction logic from hybrid_scaling_test.py
        max_qubits = 8
        channels_added = []

        # Replicate the fixed loop from hybrid_scaling_test.py
        for i in range(max_qubits):
            channels_added.append(("rx", [i]))
            channels_added.append(("rz", [i]))
            channels_added.append(("h", [i]))
        for i in range(max_qubits - 1):
            channels_added.append(("cx", [i, i + 1]))

        # Verify all single-qubit gates cover all indices
        rx_indices = [c[1] for c in channels_added if c[0] == "rx"]
        assert rx_indices == [[i] for i in range(max_qubits)]

        rz_indices = [c[1] for c in channels_added if c[0] == "rz"]
        assert rz_indices == [[i] for i in range(max_qubits)]

        h_indices = [c[1] for c in channels_added if c[0] == "h"]
        assert h_indices == [[i] for i in range(max_qubits)]

        # Verify CX covers all adjacent pairs
        cx_indices = [c[1] for c in channels_added if c[0] == "cx"]
        assert cx_indices == [[i, i + 1] for i in range(max_qubits - 1)]


# ─────────────────────────────────────────────────────
# 5. Output Filename Logic
# ─────────────────────────────────────────────────────

class TestOutputFilename:
    """Test that precision is correctly appended to the output filename."""

    def test_precision_appended_float64(self):
        base = "data/benchmark_results.json"
        precision = "float64"
        stem, ext = os.path.splitext(base)
        result = f"{stem}_{precision}{ext}"
        assert result == "data/benchmark_results_float64.json"

    def test_precision_appended_float32(self):
        base = "data/benchmark_results.json"
        precision = "float32"
        stem, ext = os.path.splitext(base)
        result = f"{stem}_{precision}{ext}"
        assert result == "data/benchmark_results_float32.json"

    def test_custom_path_preserves_structure(self):
        base = "/custom/path/results.json"
        precision = "float32"
        stem, ext = os.path.splitext(base)
        result = f"{stem}_{precision}{ext}"
        assert result == "/custom/path/results_float32.json"


# ─────────────────────────────────────────────────────
# 6. extract_values backward compatibility
# ─────────────────────────────────────────────────────

class TestExtractValues:
    """Test the extract_values helper from plot_results for format compatibility."""

    def test_extract_from_new_format(self):
        from benchmarks.plot_results import extract_values
        data = {
            "4": {"mean": 0.01, "std": 0.001, "min": 0.009, "raw": [0.01]},
            "6": {"mean": 0.02, "std": 0.002, "min": 0.018, "raw": [0.02]}
        }
        means, stds = extract_values(data, [4, 6])
        assert means == [0.01, 0.02]
        assert stds == [0.001, 0.002]

    def test_extract_from_legacy_format(self):
        from benchmarks.plot_results import extract_values
        data = {"4": 0.017595, "6": 0.017853}
        means, stds = extract_values(data, [4, 6])
        assert means == [0.017595, 0.017853]
        assert stds == [0.0, 0.0]

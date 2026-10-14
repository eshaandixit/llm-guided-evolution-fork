import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_improved


def test_quantum_result_text_is_read_and_fitness_set(tmp_path, monkeypatch):
    gene_id = "gene_success"
    results_dir = tmp_path / "sota" / "QuantumVQC" / "results"
    results_dir.mkdir(parents=True)
    (results_dir / f"{gene_id}_results.txt").write_text("0.14, 56, 0.48")
    evaluation_dir = tmp_path / "run_job_outputs" / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "slurm-123.out").write_text("job done\n")

    monkeypatch.setattr(run_improved, "SOTA_ROOT", str(results_dir.parent))
    monkeypatch.setattr(run_improved, "SLURM_OUTPUT_PATH",
                        str(tmp_path / "run_job_outputs") + "/")
    run_improved.GLOBAL_DATA[gene_id] = {
        "results_job": "123", "local_output": None, "status": "running eval",
    }

    run_improved.check4results(gene_id)

    assert run_improved.GLOBAL_DATA[gene_id]["fitness"] == (0.14, 56.0)
    assert run_improved.GLOBAL_DATA[gene_id]["status"] == "completed"


def test_missing_quantum_results_become_invalid_fitness(tmp_path, monkeypatch):
    gene_id = "gene_missing"
    results_dir = tmp_path / "sota" / "QuantumVQC" / "results"
    results_dir.mkdir(parents=True)
    evaluation_dir = tmp_path / "run_job_outputs" / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "slurm-456.out").write_text("job done\n")

    monkeypatch.setattr(run_improved, "SOTA_ROOT", str(results_dir.parent))
    monkeypatch.setattr(run_improved, "SLURM_OUTPUT_PATH",
                        str(tmp_path / "run_job_outputs") + "/")
    run_improved.GLOBAL_DATA[gene_id] = {
        "results_job": "456", "local_output": None, "status": "running eval",
    }

    run_improved.check4results(gene_id)

    assert run_improved.GLOBAL_DATA[gene_id]["fitness"] == run_improved.INVALID_FITNESS_MAX
    assert run_improved.GLOBAL_DATA[gene_id]["status"] == "completed"

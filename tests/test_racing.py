import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import racing


def write_metrics(results_dir, gene_id, accuracies, gates=28, first_seed=42, suffix=""):
    starts = [{"seed": first_seed + i, "val_accuracy": acc, "val_cross_entropy": 0.5,
               "train_accuracy": acc, "test_accuracy": acc, "weights": [float(i)]}
              for i, acc in enumerate(accuracies)]
    metrics = {"gene_id": gene_id, "obj2_gates": float(gates), "starts": starts,
               "n_starts": len(starts), "val_cross_entropy": 0.5}
    (results_dir / f"{gene_id}{suffix}_metrics.json").write_text(json.dumps(metrics))


def test_ranks_put_the_pareto_front_first():
    points = [(0.1, 30), (0.2, 20), (0.3, 30), (0.4, 40)]
    assert racing.nondominated_ranks(points) == [0, 0, 1, 2]


def test_clear_winner_and_clear_loser_need_no_more_starts():
    stats = {
        "winner": (0.05, 0.02, 3, 20),   # best on both objectives by a wide margin
        "loser": (0.60, 0.02, 3, 40),
    }
    assert racing.uncertain_genes(stats) == []


def test_close_call_on_the_front_is_uncertain():
    # "a" and "b" have the same gate count, so whichever has lower error dominates.
    # Their errors are within noise, so each one's rank depends on luck.
    stats = {"a": (0.20, 0.08, 3, 28), "b": (0.22, 0.08, 3, 28), "far": (0.70, 0.05, 3, 60)}
    assert set(racing.uncertain_genes(stats)) == {"a", "b"}


def test_genes_at_the_cap_are_not_returned():
    stats = {"a": (0.20, 0.08, 10, 28), "b": (0.22, 0.08, 3, 28)}
    assert racing.uncertain_genes(stats, max_starts=10) == ["b"]


def test_zero_sd_borrows_the_population_median():
    # Three identical accuracies give sd 0; without the floor this gene would look certain.
    stats = {"a": (0.20, 0.0, 3, 28), "b": (0.22, 0.08, 3, 28), "c": (0.21, 0.08, 3, 28)}
    assert "a" in racing.uncertain_genes(stats)


def test_merge_combines_extra_starts_and_rewrites_results(tmp_path):
    write_metrics(tmp_path, "g", [0.6, 0.7, 0.8])
    write_metrics(tmp_path, "g", [0.9, 0.9], first_seed=45, suffix="_starts3")

    fitness = racing.merge_starts(tmp_path, "g")

    expected_error = 1 - (0.6 + 0.7 + 0.8 + 0.9 + 0.9) / 5
    assert abs(fitness[0] - expected_error) < 1e-9 and fitness[1] == 28
    merged = json.loads((tmp_path / "g_metrics.json").read_text())
    assert merged["n_starts"] == 5
    assert [s["seed"] for s in merged["starts"]] == [42, 43, 44, 45, 46]
    assert merged["trained_weights"] == [0.0]          # first start of the extra file, acc 0.9
    obj1, obj2, _ = (float(v) for v in (tmp_path / "g_results.txt").read_text().split(","))
    assert abs(obj1 - expected_error) < 1e-9 and obj2 == 28


def test_merging_twice_gives_the_same_result(tmp_path):
    write_metrics(tmp_path, "g", [0.6, 0.7, 0.8])
    write_metrics(tmp_path, "g", [0.9], first_seed=45, suffix="_starts3")
    first = racing.merge_starts(tmp_path, "g")
    second = racing.merge_starts(tmp_path, "g")
    assert first == second
    assert len(racing.collect_starts(tmp_path, "g")) == 4


def test_race_population_adds_starts_only_where_needed(tmp_path, monkeypatch):
    import run_improved
    from deap import creator

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    write_metrics(results_dir, "close_a", [0.80, 0.70, 0.90], gates=28)
    write_metrics(results_dir, "close_b", [0.75, 0.85, 0.70], gates=28)
    write_metrics(results_dir, "far", [0.30, 0.35, 0.30], gates=60)
    population = []
    for gene_id in ("close_a", "close_b", "far"):
        ind = creator.Individual([gene_id])
        ind.fitness.values = racing.summarize(racing.collect_starts(results_dir, gene_id))[0], 28.0
        population.append(ind)

    submitted = []

    def fake_submit(gene_id, start_offset, n_starts):
        submitted.append((gene_id, start_offset, n_starts))
        write_metrics(results_dir, gene_id, [0.8] * n_starts, first_seed=42 + start_offset,
                      suffix=f"_starts{start_offset}")
        return {"local_output": "job done"}

    monkeypatch.setattr(run_improved, "RACING", True)
    monkeypatch.setattr(run_improved, "SOTA_ROOT", str(tmp_path))
    monkeypatch.setattr(run_improved, "submit_extra_starts", fake_submit)
    monkeypatch.setattr(run_improved, "GLOBAL_DATA", {g: {} for g in ("close_a", "close_b", "far")})

    run_improved.race_population(population, loop_delay=0)

    raced = {gene_id for gene_id, _, _ in submitted}
    assert raced == {"close_a", "close_b"}
    assert all(offset >= 3 for _, offset, _ in submitted)
    for gene_id in raced:
        n = len(racing.collect_starts(results_dir, gene_id))
        assert n <= 10
        assert run_improved.GLOBAL_DATA[gene_id]["racing_starts"] == n
    assert run_improved.GLOBAL_DATA["far"]["racing_starts"] == 3

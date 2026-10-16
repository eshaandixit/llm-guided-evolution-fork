import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import racing_analysis as ra


def test_hypervolume_of_simple_fronts():
    assert ra.hypervolume([(0.5, 28)], ref_error=1.0, ref_gates=56) == 0.5 * 28
    # Two points: (0.2, 40) and (0.4, 20). Area = 0.8*16 + 0.6*20.
    assert abs(ra.hypervolume([(0.4, 20), (0.2, 40)], 1.0, 56) - (0.8 * 16 + 0.6 * 20)) < 1e-12
    # Dominated and out-of-range points add nothing.
    assert ra.hypervolume([(0.4, 20), (0.5, 30), (1.2, 10)], 1.0, 56) == 0.6 * 36


def write_run(exp, name, entries):
    path = exp / name / "checkpoints"
    path.mkdir(parents=True)
    with open(path / "progress.jsonl", "w") as f:
        for gen, starts, front in entries:
            f.write(json.dumps({"generation": gen, "training_starts": starts, "front": front}) + "\n")


def test_runs_are_compared_at_the_control_budget(tmp_path):
    write_run(tmp_path, "control_1", [(0, 160, [["c0", 0.33, 28, 10]]),
                                      (1, 300, [["c1", 0.30, 28, 10]])])
    write_run(tmp_path, "racing_1", [(0, 48, [["r0", 0.33, 28, 3]]),
                                     (1, 150, [["r1", 0.25, 28, 6]]),
                                     (2, 290, [["r2", 0.20, 26, 7]]),
                                     (3, 420, [["r3", 0.10, 20, 9]])])   # past the budget
    rescore = tmp_path / "rescore"
    rescore.mkdir()
    for gene, acc in (("r2", 0.75), ("c1", 0.70)):
        (rescore / f"{gene}_starts1000_metrics.json").write_text(
            json.dumps({"starts": [{"val_accuracy": acc}] * 30}))

    report = ra.analyse(str(tmp_path))
    assert report["budget_starts"] == 300
    racing = report["runs"]["racing_1"]
    assert racing["generation_at_budget"] == 2                     # generation 3 used too much
    assert abs(racing["hv_at_budget"] - 0.8 * 30) < 1e-9
    assert racing["starts_to_reach_control_hv"] == 150             # 0.75*28 >= control's 0.70*28
    assert abs(racing["mean_reported_minus_true_error"] - (0.20 - 0.25)) < 1e-9
    assert abs(report["runs"]["control_1"]["hv_at_budget_rescored"] - 0.70 * 28) < 1e-9
    assert "racing_1" in ra.summarize(report)
    ra.plot(report, str(tmp_path / "hv.png"))
    assert (tmp_path / "hv.png").stat().st_size > 0

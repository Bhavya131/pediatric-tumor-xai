"""End-to-end test on synthetic data (random data, no biological meaning)."""
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)


def test_synthetic_data_shape():
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_test_data import make
    expr, meta = make(n_samples=60, n_genes=100)
    assert expr.shape == (100, 60)
    assert set(meta["subgroup"]) <= {"G1", "G2", "G3"}


def test_full_workflow():
    run([sys.executable, "scripts/make_test_data.py"])
    run(["snakemake", "-s", "workflow/Snakefile", "--cores", "2",
         "--configfile", "config/test.yaml", "--forceall"])
    out = ROOT / "results/test"
    assert (out / "report.html").exists()

    summ = pd.read_csv(out / "models/cv_summary.csv", index_col=0)
    # A real model must clearly beat the majority-class baseline on this easy task
    best = (out / "models/best_model.txt").read_text().strip()
    assert summ.loc[best, "balanced_accuracy_mean"] > summ.loc["dummy", "balanced_accuracy_mean"] + 0.3

    sv = json.loads((out / "survival/survival_summary.json").read_text())
    assert not sv.get("skipped")
    assert sv["cindex_label_only"]["mean"] > 0.5

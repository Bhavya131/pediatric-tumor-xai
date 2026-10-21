"""Create a small synthetic expression + metadata set for tests and CI.

Three tumour 'subgroups' differ in the mean of the first 30 genes. Survival
times depend on the subgroup. The data are random and have NO biological
meaning; they only exist so the workflow can be tested end to end.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make(n_samples=150, n_genes=400, n_informative=30, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.array(["G1", "G2", "G3"])
    labels = rng.choice(classes, size=n_samples, p=[0.4, 0.35, 0.25])

    log_expr = rng.normal(8, 1.5, size=(n_genes, n_samples))
    shifts = {"G1": 0.0, "G2": 1.2, "G3": -1.2}
    gene_signs = rng.choice([-1, 1], size=n_informative)
    for j, lab in enumerate(labels):
        log_expr[:n_informative, j] += shifts[lab] * gene_signs
    counts = np.round(2 ** log_expr).astype(int)  # raw-count-like values

    samples = [f"S{i:03d}" for i in range(n_samples)]
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    expr = pd.DataFrame(counts, index=genes, columns=samples)
    expr.index.name = "gene"

    scale = {"G1": 70, "G2": 40, "G3": 20}
    time = np.array([rng.exponential(scale[l]) for l in labels])
    censor = 80.0
    event = (time < censor).astype(int)
    time = np.minimum(time, censor)
    meta = pd.DataFrame({
        "sample_id": samples,
        "subgroup": labels,
        "os_time": time.round(2),
        "os_event": event,
    })
    return expr, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    expr, meta = make(seed=args.seed)
    expr.to_csv(out / "expression.csv")
    meta.to_csv(out / "metadata.csv", index=False)
    print(f"Wrote synthetic data to {out}")

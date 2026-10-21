"""Load, clean and QC the expression matrix.

Only unsupervised steps (no labels) are used for filtering, so this step
cannot leak label information into the model evaluation.
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from common import load_config, ensure_dir


def main(cfg_path):
    cfg = load_config(cfg_path)
    out = ensure_dir(f"{cfg['outdir']}/prepared")

    expr = pd.read_csv(cfg["expression"], index_col=0)  # genes x samples
    meta = pd.read_csv(cfg["metadata"])
    sid, lab = cfg["sample_col"], cfg["label_col"]

    # Duplicate gene ids (e.g. several probes) -> average
    if expr.index.duplicated().any():
        expr = expr.groupby(level=0).mean()

    # Align samples between expression and metadata
    meta = meta.drop_duplicates(sid).set_index(sid)
    common = [s for s in expr.columns if s in meta.index]
    if len(common) < 20:
        raise SystemExit(f"Only {len(common)} samples overlap; check sample ids.")
    expr, meta = expr[common], meta.loc[common]

    # Drop samples without a label and very small classes
    meta = meta[meta[lab].notna()]
    counts = meta[lab].value_counts()
    keep_classes = counts[counts >= 6].index
    dropped_classes = sorted(set(counts.index) - set(keep_classes))
    meta = meta[meta[lab].isin(keep_classes)]
    expr = expr[meta.index]

    # Missing values: drop genes with >20% NaN, median-impute the rest
    expr = expr.loc[expr.isna().mean(axis=1) <= 0.2]
    expr = expr.apply(lambda r: r.fillna(r.median()), axis=1)

    # Log-transform raw-count-like data (heuristic: max value > 100)
    log_applied = bool(expr.values.max() > 100)
    if log_applied:
        expr = np.log2(expr.clip(lower=0) + 1)

    # Unsupervised variance filter
    n_top = min(int(cfg["n_top_genes"]), expr.shape[0])
    top = expr.var(axis=1).sort_values(ascending=False).index[:n_top]
    expr = expr.loc[top]

    X = expr.T  # samples x genes
    X.to_csv(out / "X.csv")
    y = meta[[lab]].rename(columns={lab: "label"})
    y.to_csv(out / "y.csv")

    surv_cols = [c for c in (cfg.get("time_col"), cfg.get("event_col")) if c]
    if len(surv_cols) == 2 and all(c in meta.columns for c in surv_cols):
        meta[surv_cols].rename(columns={cfg["time_col"]: "time",
                                        cfg["event_col"]: "event"}).to_csv(out / "survival.csv")

    # PCA QC plot coloured by label (QC only - not used for modelling)
    pcs = PCA(n_components=2, random_state=cfg["seed"]).fit(X.values)
    coords = pcs.transform(X.values)
    fig, ax = plt.subplots(figsize=(5, 4))
    for cls in sorted(y["label"].unique()):
        m = (y["label"] == cls).values
        ax.scatter(coords[m, 0], coords[m, 1], s=14, label=str(cls), alpha=0.8)
    ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]:.0%})")
    ax.set_title("PCA of filtered expression")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pca.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_samples": int(X.shape[0]),
        "n_genes_after_filter": int(X.shape[1]),
        "class_counts": {str(k): int(v) for k, v in y["label"].value_counts().items()},
        "log2_transform_applied": log_applied,
        "dropped_small_classes": [str(c) for c in dropped_classes],
    }
    (out / "qc_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)

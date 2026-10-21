"""Nested, repeated cross-validation of baseline and ML models.

Outer loop: estimates performance. Inner loop: tunes hyper-parameters (and the
number of selected genes) using ONLY the outer training data.
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from common import load_config, ensure_dir
from models import build_models


def safe_auc(y_true, proba, classes):
    try:
        if len(classes) == 2:
            return roc_auc_score(y_true, proba[:, 1])
        return roc_auc_score(y_true, proba, multi_class="ovr", labels=classes)
    except ValueError:
        return np.nan


def main(cfg_path):
    cfg = load_config(cfg_path)
    out = ensure_dir(f"{cfg['outdir']}/models")
    X = pd.read_csv(f"{cfg['outdir']}/prepared/X.csv", index_col=0)
    y = pd.read_csv(f"{cfg['outdir']}/prepared/y.csv", index_col=0)["label"]
    seed = cfg["seed"]
    models = build_models(X.shape[1], seed)

    outer = RepeatedStratifiedKFold(n_splits=cfg["outer_folds"],
                                    n_repeats=cfg["outer_repeats"], random_state=seed)
    rows = []
    for i, (tr, te) in enumerate(outer.split(X, y)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        for name, (pipe, grid) in models.items():
            inner = StratifiedKFold(cfg["inner_folds"], shuffle=True, random_state=seed)
            gs = GridSearchCV(pipe, grid, cv=inner, scoring="balanced_accuracy", n_jobs=1)
            gs.fit(Xtr, ytr)
            pred = gs.predict(Xte)
            auc = safe_auc(yte, gs.predict_proba(Xte), list(gs.classes_))
            rows.append({"outer_split": i, "model": name,
                         "balanced_accuracy": balanced_accuracy_score(yte, pred),
                         "macro_f1": f1_score(yte, pred, average="macro"),
                         "roc_auc_ovr": auc,
                         "best_params": json.dumps(gs.best_params_, default=str)})
        print(f"outer split {i + 1} done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out / "cv_results.csv", index=False)
    summary = (res.groupby("model")[["balanced_accuracy", "macro_f1", "roc_auc_ovr"]]
                  .agg(["mean", "std"]).round(3))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.sort_values("balanced_accuracy_mean", ascending=False)
    summary.to_csv(out / "cv_summary.csv")

    candidates = summary.drop(index="dummy", errors="ignore")
    best = candidates.index[0]
    (out / "best_model.txt").write_text(best)
    print(summary)
    print("Best non-baseline model:", best)

    fig, ax = plt.subplots(figsize=(6, 4))
    order = list(summary.index)
    ax.boxplot([res.loc[res.model == m, "balanced_accuracy"] for m in order])
    ax.set_xticklabels(order)
    ax.set_ylabel("Balanced accuracy (outer folds)")
    ax.set_title("Nested CV performance")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "cv_boxplot.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)

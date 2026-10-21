"""Survival analysis: Kaplan-Meier by label + cross-validated Cox models.

Skipped gracefully if no survival columns are configured/available.
"""
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from common import load_config, ensure_dir


def cv_cindex(features, surv, n_splits, seed, penalizer=0.1):
    scores = []
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(features):
        ftr, fte = features.iloc[tr].copy(), features.iloc[te].copy()
        keep = ftr.columns[ftr.std() > 0]
        mu, sd = ftr[keep].mean(), ftr[keep].std()
        ftr, fte = (ftr[keep] - mu) / sd, (fte[keep] - mu) / sd
        dtr = ftr.join(surv.iloc[tr][["time", "event"]])
        try:
            cph = CoxPHFitter(penalizer=penalizer).fit(dtr, "time", "event")
            risk = cph.predict_partial_hazard(fte).values.ravel()
            scores.append(concordance_index(surv.iloc[te]["time"], -risk,
                                            surv.iloc[te]["event"]))
        except Exception as exc:
            print(f"Cox fit failed in a fold: {exc}")
    return scores


def main(cfg_path):
    cfg = load_config(cfg_path)
    out = ensure_dir(f"{cfg['outdir']}/survival")
    surv_path = f"{cfg['outdir']}/prepared/survival.csv"
    try:
        surv = pd.read_csv(surv_path, index_col=0)
    except FileNotFoundError:
        (out / "survival_summary.json").write_text(json.dumps({"skipped": True}))
        print("No survival data configured - skipping.")
        return

    X = pd.read_csv(f"{cfg['outdir']}/prepared/X.csv", index_col=0)
    y = pd.read_csv(f"{cfg['outdir']}/prepared/y.csv", index_col=0)["label"]
    surv = surv.loc[X.index].dropna()
    surv = surv[surv["time"] > 0]
    X, y = X.loc[surv.index], y.loc[surv.index]

    # Kaplan-Meier by label
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for cls in sorted(y.unique()):
        m = (y == cls).values
        KaplanMeierFitter().fit(surv.loc[m, "time"], surv.loc[m, "event"],
                                label=f"{cls} (n={m.sum()})").plot_survival_function(ax=ax, ci_show=False)
    lr = multivariate_logrank_test(surv["time"], y, surv["event"])
    ax.set_title(f"Overall survival by label (log-rank p = {lr.p_value:.2g})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    fig.tight_layout()
    fig.savefig(out / "km_by_label.png", dpi=150)
    plt.close(fig)

    # Cross-validated C-index: label-only vs top-variance genes (no labels used
    # for gene selection, so no leakage).
    warnings.filterwarnings("ignore")
    label_feats = pd.get_dummies(y, drop_first=True).astype(float)
    top_genes = X.var().sort_values(ascending=False).index[:20]
    k = 5
    res = {
        "n_samples": int(len(surv)),
        "logrank_p": float(lr.p_value),
        "cindex_label_only": cv_cindex(label_feats, surv, k, cfg["seed"]),
        "cindex_top20_variance_genes": cv_cindex(X[top_genes], surv, k, cfg["seed"]),
    }
    for key in ("cindex_label_only", "cindex_top20_variance_genes"):
        vals = res[key]
        res[key] = {"mean": float(np.mean(vals)) if vals else None,
                    "sd": float(np.std(vals)) if vals else None, "folds": vals}
    (out / "survival_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=2))
    print("C-index label only:", res["cindex_label_only"]["mean"],
          "| top-20 genes:", res["cindex_top20_variance_genes"]["mean"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)

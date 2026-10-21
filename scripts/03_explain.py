"""Explain the best model on a held-out test split.

Reports permutation importance and (where supported) SHAP values for the genes
that survive feature selection. Explanations describe what the MODEL uses, not
causal biology - see the README for caveats.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from common import load_config, ensure_dir
from models import build_models


def mean_abs_shap(clf, X_sel, model_name):
    """Return mean |SHAP| per feature, or None if unsupported."""
    import shap
    try:
        if model_name == "logreg":
            expl = shap.LinearExplainer(clf, X_sel)
        else:
            expl = shap.TreeExplainer(clf)
        sv = expl.shap_values(X_sel)
    except Exception as exc:  # unsupported model / version differences
        print(f"SHAP skipped: {exc}")
        return None
    if isinstance(sv, list):                    # list of (n, p) per class
        arr = np.stack([np.abs(s) for s in sv])  # (c, n, p)
        return arr.mean(axis=(0, 1))
    sv = np.asarray(sv)
    if sv.ndim == 3:                            # (n, p, c)
        return np.abs(sv).mean(axis=(0, 2))
    return np.abs(sv).mean(axis=0)


def main(cfg_path, model_override=None):
    cfg = load_config(cfg_path)
    out = ensure_dir(f"{cfg['outdir']}/explain")
    X = pd.read_csv(f"{cfg['outdir']}/prepared/X.csv", index_col=0)
    y = pd.read_csv(f"{cfg['outdir']}/prepared/y.csv", index_col=0)["label"]
    name = model_override or open(f"{cfg['outdir']}/models/best_model.txt").read().strip()
    seed = cfg["seed"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    pipe, grid = build_models(X.shape[1], seed)[name]
    gs = GridSearchCV(pipe, grid, scoring="balanced_accuracy",
                      cv=StratifiedKFold(cfg["inner_folds"], shuffle=True, random_state=seed))
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    pred = best.predict(Xte)
    bal = balanced_accuracy_score(yte, pred)
    print(f"{name}: held-out balanced accuracy = {bal:.3f}")

    # Features that survive scaling + selection
    scaler, selector, clf = best.named_steps["scale"], best.named_steps["select"], best.named_steps["clf"]
    kept = X.columns[selector.get_support()]
    Xte_sel = pd.DataFrame(selector.transform(scaler.transform(Xte)), columns=kept, index=Xte.index)

    perm = permutation_importance(clf, Xte_sel.values, yte, scoring="balanced_accuracy",
                                  n_repeats=20, random_state=seed, n_jobs=1)
    imp = pd.DataFrame({"gene": kept, "perm_importance": perm.importances_mean,
                        "perm_std": perm.importances_std})
    shap_vals = mean_abs_shap(clf, Xte_sel.values, name)
    if shap_vals is not None:
        imp["mean_abs_shap"] = shap_vals
    imp = imp.sort_values("perm_importance", ascending=False)
    imp.to_csv(out / "feature_importance.csv", index=False)

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top["gene"], top["perm_importance"], xerr=top["perm_std"])
    ax.set_xlabel("Drop in balanced accuracy when permuted")
    ax.set_title(f"Top genes used by {name} (held-out data)")
    fig.tight_layout()
    fig.savefig(out / "top_genes_permutation.png", dpi=150)
    plt.close(fig)

    if shap_vals is not None:
        top = imp.sort_values("mean_abs_shap", ascending=False).head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top["gene"], top["mean_abs_shap"])
        ax.set_xlabel("mean |SHAP value|")
        ax.set_title(f"SHAP importance ({name})")
        fig.tight_layout()
        fig.savefig(out / "top_genes_shap.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(yte, pred, ax=ax, colorbar=False)
    ax.set_title(f"Held-out confusion matrix\nbalanced acc. = {bal:.2f}")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    (out / "explain_summary.txt").write_text(
        f"model={name}\nheld_out_balanced_accuracy={bal:.4f}\nn_test={len(yte)}\n"
        f"best_params={gs.best_params_}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="override best model (for testing)")
    a = ap.parse_args()
    main(a.config, a.model)

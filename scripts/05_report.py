"""Assemble a single self-contained HTML report from the pipeline outputs."""
import argparse
import base64
import html
import json
from pathlib import Path
import pandas as pd
from common import load_config


def img(path):
    p = Path(path)
    if not p.exists():
        return "<p><em>(figure not available)</em></p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:560px;width:100%">'


def main(cfg_path):
    cfg = load_config(cfg_path)
    o = Path(cfg["outdir"])
    qc = json.loads((o / "prepared/qc_summary.json").read_text())
    summ = pd.read_csv(o / "models/cv_summary.csv", index_col=0)
    best = (o / "models/best_model.txt").read_text().strip()
    expl = (o / "explain/explain_summary.txt").read_text()
    imp = pd.read_csv(o / "explain/feature_importance.csv").head(15)
    sv = json.loads((o / "survival/survival_summary.json").read_text())

    surv_html = "<p>No survival data configured.</p>"
    if not sv.get("skipped"):
        a, b = sv["cindex_label_only"], sv["cindex_top20_variance_genes"]
        surv_html = (
            f"<p>n = {sv['n_samples']}, log-rank p = {sv['logrank_p']:.3g}.</p>"
            f"<p>5-fold CV concordance index: label only = {a['mean']:.3f} &plusmn; {a['sd']:.3f}; "
            f"top-20 variance genes = {b['mean']:.3f} &plusmn; {b['sd']:.3f}. "
            "A C-index of 0.5 is random ordering.</p>"
            + img(o / "survival/km_by_label.png"))

    shap_fig = img(o / "explain/top_genes_shap.png") if (o / "explain/top_genes_shap.png").exists() else ""
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Explainable ML on tumour expression - report</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;line-height:1.5}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}h2{{border-bottom:1px solid #ddd;padding-bottom:4px}}</style></head><body>
<h1>Explainable ML on tumour expression data</h1>
<h2>1. Data and QC</h2>
<p>{qc['n_samples']} samples, {qc['n_genes_after_filter']} genes after unsupervised variance filtering.
log2 transform applied: {qc['log2_transform_applied']}. Class counts: {html.escape(json.dumps(qc['class_counts']))}.</p>
{img(o / 'prepared/pca.png')}
<h2>2. Model comparison (nested repeated CV)</h2>
{summ.to_html(float_format=lambda v: f'{v:.3f}')}
<p>Best non-baseline model: <b>{html.escape(best)}</b>. The <code>dummy</code> row is a majority-class baseline;
a model that does not clearly beat it has learned nothing useful.</p>
{img(o / 'models/cv_boxplot.png')}
<h2>3. Explanation of the best model (held-out split)</h2>
<pre>{html.escape(expl)}</pre>
{img(o / 'explain/confusion_matrix.png')}
{img(o / 'explain/top_genes_permutation.png')}
{shap_fig}
{imp.to_html(index=False, float_format=lambda v: f'{v:.4f}')}
<h2>4. Survival</h2>
{surv_html}
<h2>Limitations</h2>
<ul>
<li>Feature importance shows what the model uses, not causal biology; correlated genes can share or swap importance.</li>
<li>Performance is estimated within one cohort. Without an independent external cohort, generalisation is unproven.</li>
<li>Batch effects and cohort-specific preprocessing can inflate accuracy; check them before trusting results.</li>
<li>Research use only; nothing here is a diagnostic tool.</li></ul>
</body></html>"""
    (o / "report.html").write_text(page, encoding="utf-8")
    print(f"Wrote {o / 'report.html'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)

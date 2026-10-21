# Explainable ML on tumour expression data

[![CI](https://github.com/<your-username>/pediatric-tumor-xai/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/pediatric-tumor-xai/actions)

A reproducible, leakage-safe workflow that asks: **can a transparent, well-validated model predict tumour
subtype (and relate to survival) from bulk gene-expression data, and which genes drive its predictions?**

It is written to be run on public paediatric tumour cohorts (see [`data/README.md`](data/README.md)); the
included test dataset is synthetic and only checks that the code works.

> **Status:** workflow tested end to end on synthetic data.
> Real-data results: _add here once you have run it (dataset, accession, n, headline numbers, one figure)._

## What it does

```
expression + metadata
   -> 01 prepare   dedupe genes, align samples, log2 (if counts), unsupervised variance filter, PCA QC
   -> 02 train     nested repeated CV: majority-class baseline, logistic regression, random forest,
                   gradient boosting (scaling + gene selection inside each fold)
   -> 03 explain   held-out split: permutation importance + SHAP, confusion matrix
   -> 04 survival  Kaplan-Meier by label, cross-validated Cox C-index (label-only vs top-variance genes)
   -> 05 report    one self-contained HTML report
```

Design choices that matter:
- **No information leakage:** scaling and supervised gene selection are *inside* the cross-validation pipeline;
  the only filtering done beforehand is unsupervised (variance), which uses no labels.
- **Baselines first:** every model is compared with a majority-class dummy; the report shows both.
- **Nested CV:** hyper-parameters (and the number of genes) are tuned on the training data of each outer fold only.
- **Honest explanations:** importance scores describe what the model uses, not causal biology (see limitations).

## Quick start

```bash
git clone https://github.com/<your-username>/pediatric-tumor-xai && cd pediatric-tumor-xai
conda env create -f environment.yml && conda activate pediatric-tumor-xai   # or: pip install -r requirements.txt

# smoke test on synthetic data (~1 min)
python scripts/make_test_data.py
snakemake --cores 2 --configfile config/test.yaml
# open results/test/report.html

# real data: place expression.csv + metadata.csv in data/raw/, edit config/config.yaml, then
snakemake --cores 4
pytest -q          # runs the same end-to-end test used in CI
```

## Repository layout

```
workflow/Snakefile      pipeline definition          config/   run settings (real + test)
scripts/                one script per step          tests/    end-to-end test (also run in CI)
data/                   input format + test data     results/  generated outputs (git-ignored)
```

## Limitations

- Performance is estimated within a single cohort; independent external validation is needed before any claim of
  generalisation.
- Batch effects and platform differences can inflate accuracy; check them for each dataset.
- SHAP/permutation importance are model explanations, not evidence of biological causality; correlated genes can
  share importance.
- Research code only; not a diagnostic tool. Use only open, de-identified public data.

## Possible extensions
- Tumour microenvironment estimates (immune/stromal deconvolution) and their association with subtype/outcome.
- Train on one cohort and test on another (external validation).
- Port the workflow to Nextflow/nf-core conventions.

## Author
Bhavyashree Vishwanatha, MSc Bioinformatics, Saarland University. Licensed under MIT.

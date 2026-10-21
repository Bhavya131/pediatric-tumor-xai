"""Model definitions shared by the evaluation and explanation steps."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(n_features, seed):
    """Return {name: (pipeline, param_grid)}.

    Scaling and feature selection live INSIDE the pipeline, so they are
    refitted on the training part of every CV fold (no information leakage).
    """
    ks = sorted({min(50, n_features), min(200, n_features)})

    def pipe(clf):
        return Pipeline([("scale", StandardScaler()),
                         ("select", SelectKBest(f_classif)),
                         ("clf", clf)])

    return {
        "dummy": (pipe(DummyClassifier(strategy="most_frequent")),
                  {"select__k": [ks[0]]}),
        "logreg": (pipe(LogisticRegression(max_iter=3000, class_weight="balanced")),
                   {"select__k": ks, "clf__C": [0.01, 0.1, 1.0]}),
        "random_forest": (pipe(RandomForestClassifier(
                              n_estimators=300, class_weight="balanced",
                              random_state=seed, n_jobs=-1)),
                          {"select__k": ks, "clf__max_depth": [None, 6]}),
        "hist_gb": (pipe(HistGradientBoostingClassifier(random_state=seed)),
                    {"select__k": ks, "clf__learning_rate": [0.05, 0.1],
                     "clf__max_depth": [3]}),
    }

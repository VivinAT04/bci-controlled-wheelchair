"""Evaluation protocols and metrics for the EEG classifiers."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def evaluate_cv(clf, X, y, n_splits: int = 5, seed: int = 42) -> dict:
    """Within-subject stratified k-fold CV."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    return _summarise(y, y_pred)


def evaluate_heldout_session(clf, X, y, meta) -> dict:
    """Train on session 'T', test on session 'E' — the standard 2a benchmark."""
    sessions = meta["session"].values
    uniq = sorted(np.unique(sessions))
    train_s, test_s = uniq[0], uniq[-1]
    tr, te = sessions == train_s, sessions == test_s
    clf.fit(X[tr], y[tr])
    y_pred = clf.predict(X[te])
    return _summarise(y[te], y_pred)


def _summarise(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.unique(y_true)),
    }

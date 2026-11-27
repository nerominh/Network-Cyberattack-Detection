"""Shared evaluation for the 72 models, so both partners log results in the same format.

    import sys; sys.path.append("..")
    from src.evaluation import evaluate, plot_sweep, attack_recall, RESULTS
"""
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

RESULTS = Path(__file__).resolve().parents[1] / "results"


def scores(y_true, y_pred, prefix):
    # Precision, recall, F1 (attack = positive class) and false positive rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_fpr": fp / (fp + tn),
    }


def evaluate(pipeline, model, transformation, param, value, X_train, y_train, X_val, y_val):
    """Fit one pipeline on train, score it on train and validation, return one results row."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - start

    row = {"model": model, "transformation": transformation, "param": param, "value": value}
    row.update(scores(y_train, pipeline.predict(X_train), "train"))
    row.update(scores(y_val, pipeline.predict(X_val), "val"))
    row["fit_time_s"] = round(fit_time, 2)
    row["converged"] = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return row


def plot_sweep(results, title, save_to=None):
    """Left: train (dashed) vs validation (solid) F1. Right: validation precision, recall and FPR."""
    param = results["param"].iloc[0]
    direction = "larger = stronger regularization" if param == "alpha" else "smaller = stronger regularization"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(results["value"], results["train_f1"], "--o", label="train F1")
    axes[0].plot(results["value"], results["val_f1"], "-o", label="validation F1")
    axes[0].set_title(f"{title}: train vs validation F1")
    for metric in ["val_precision", "val_recall", "val_fpr"]:
        axes[1].plot(results["value"], results[metric], "-o", label=metric.replace("val_", ""))
    axes[1].set_title(f"{title}: validation metrics")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(f"{param} ({direction})")
        ax.legend()
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=150)
    plt.show()


def attack_recall(y_true, y_pred, attack_cat):
    """For each attack type: how many rows, how many the model caught and missed, and the recall."""
    df = pd.DataFrame({"type": attack_cat.values, "true": y_true.values, "pred": y_pred})
    attacks = df[df["true"] == 1]
    out = attacks.groupby("type").agg(rows=("pred", "size"), caught=("pred", "sum"))
    out["missed"] = out["rows"] - out["caught"]
    out["recall"] = (out["caught"] / out["rows"]).round(3)
    return out.sort_values("recall")

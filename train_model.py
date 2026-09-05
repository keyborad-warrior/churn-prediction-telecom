"""
Trains and compares three churn-prediction models (Logistic Regression,
Random Forest, Gradient Boosting) on the synthetic dataset, then saves
the visual artifacts referenced in the Week 4 report:
  - outputs/figures/chart_model_comparison.png
  - outputs/figures/chart_feature_importance.png
  - outputs/metrics.json  (accuracy / precision / recall + confusion matrix)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

NAVY, ORANGE, TEAL, GREY = "#1F3A5F", "#E07A2C", "#2C8C8C", "#B0B0B0"

CATEGORICAL = ["contract", "tech_support", "internet_service", "payment_method", "paperless_billing"]
NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_tickets"]


def build_pipeline(model, scale_numeric=False):
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", numeric_step, NUMERIC),
        ],
    )
    return Pipeline([("pre", pre), ("model", model)])


def get_feature_names(pipeline):
    pre = pipeline.named_steps["pre"]
    cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    return cat_names + NUMERIC


def model_comparison_chart(results: dict, out_path: str):
    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall"]
    x = np.arange(len(models))
    width = 0.25
    colors = [NAVY, TEAL, ORANGE]

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, metric in enumerate(metrics):
        vals = [results[m][metric] for m in models]
        ax.bar(x + (i - 1) * width, vals, width, label=metric.capitalize(), color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def feature_importance_chart(names, importances, out_path: str, top_n=8):
    order = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [NAVY if i >= top_n - 3 else GREY for i in range(top_n)]
    ax.barh(np.array(names)[order], np.array(importances)[order], color=colors)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Top Predictors of Customer Churn", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    df = pd.read_csv("data/customers.csv")
    X = df[CATEGORICAL + NUMERIC]
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.28, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": (LogisticRegression(max_iter=5000), True),
        "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=42), False),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=42), False),
    }

    results, fitted = {}, {}
    for name, (model, scale_numeric) in candidates.items():
        pipe = build_pipeline(model, scale_numeric=scale_numeric)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 3),
            "precision": round(precision_score(y_test, preds), 3),
            "recall": round(recall_score(y_test, preds), 3),
        }
        fitted[name] = pipe

    best_name = max(results, key=lambda m: results[m]["recall"])
    best_pipe = fitted[best_name]
    best_preds = best_pipe.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, best_preds).ravel()

    # Feature importance is reported from the Gradient Boosting model specifically,
    # since it's the tree-based model most likely to be recommended in the report
    # and exposes feature_importances_ directly.
    importance_source = fitted["Gradient Boosting"]
    feature_names = get_feature_names(importance_source)
    importances = importance_source.named_steps["model"].feature_importances_

    model_comparison_chart(results, "outputs/figures/chart_model_comparison.png")
    feature_importance_chart(feature_names, importances, "outputs/figures/chart_feature_importance.png")

    metrics_out = {
        "results_by_model": results,
        "best_model": best_name,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(json.dumps(metrics_out, indent=2))


if __name__ == "__main__":
    main()

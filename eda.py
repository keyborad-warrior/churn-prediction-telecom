"""
Exploratory data analysis: churn rate by contract type, and a simulated
12-month churn trend (since the synthetic dataset has no real dates,
the trend is illustrated with a seasonal multiplier for presentation
purposes only, as noted in the report).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, ORANGE, TEAL = "#1F3A5F", "#E07A2C", "#2C8C8C"


def churn_by_contract_chart(df: pd.DataFrame, out_path: str):
    rates = df.groupby("contract")["churn"].mean().reindex(
        ["Month-to-Month", "One Year", "Two Year"]
    ) * 100

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(rates.index, rates.values, color=[ORANGE, TEAL, NAVY], width=0.55)
    for b, v in zip(bars, rates.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Customer Churn Rate by Contract Type", fontweight="bold")
    ax.set_ylim(0, max(rates.values) + 10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return rates


def simulated_monthly_trend_chart(df: pd.DataFrame, out_path: str, seed=7):
    """Illustrative only: overlays a seasonal wave on the base churn rate
    to demonstrate how a real time-indexed dataset would be visualized."""
    rng = np.random.default_rng(seed)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    base_rate = df["churn"].mean() * 100
    seasonal = np.sin(np.linspace(0, 2 * np.pi, 12)) * 2.5
    noise = rng.normal(0, 0.4, size=12)
    trend = base_rate + seasonal + noise

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(months, trend, marker="o", color=NAVY, linewidth=2.2,
            markerfacecolor=ORANGE, markeredgecolor=ORANGE)
    ax.fill_between(months, trend, trend.min() - 1, color=NAVY, alpha=0.07)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Monthly Churn Rate Trend (Simulated Seasonality)", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return trend


if __name__ == "__main__":
    df = pd.read_csv("data/customers.csv")
    rates = churn_by_contract_chart(df, "outputs/figures/chart_churn_by_contract.png")
    print("Churn rate by contract:\n", rates.round(1))
    simulated_monthly_trend_chart(df, "outputs/figures/chart_churn_trend.png")
    print("Saved EDA charts to outputs/figures/")

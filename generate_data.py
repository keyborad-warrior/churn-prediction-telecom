"""
Generates a synthetic (hypothetical) telecom customer dataset for the
churn-prediction project described in the Week 4 capstone report.

No real customer data is used anywhere in this project — every record
is randomly generated but structured to reflect realistic churn patterns
(e.g. month-to-month contracts churn far more than two-year contracts).
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 7000


def generate_customers(n=N_CUSTOMERS, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        size=n,
        p=[0.55, 0.25, 0.20],
    )
    tenure_months = rng.integers(1, 73, size=n)
    monthly_charges = np.round(rng.normal(70, 25, size=n).clip(15, 150), 2)
    total_charges = np.round(monthly_charges * tenure_months * rng.uniform(0.9, 1.0, size=n), 2)

    tech_support = rng.choice(["Yes", "No"], size=n, p=[0.45, 0.55])
    internet_service = rng.choice(["DSL", "Fiber Optic", "No"], size=n, p=[0.35, 0.45, 0.20])
    payment_method = rng.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        size=n,
        p=[0.35, 0.20, 0.22, 0.23],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.6, 0.4])
    support_tickets = rng.poisson(1.2, size=n)

    # --- Latent churn probability, built from the same drivers used in the report ---
    base = -1.4
    contract_effect = np.select(
        [contract == "Month-to-Month", contract == "One Year", contract == "Two Year"],
        [1.6, -0.3, -1.5],
    )
    tenure_effect = -0.03 * tenure_months
    charge_effect = 0.015 * (monthly_charges - 70)
    support_effect = np.where(tech_support == "No", 0.55, -0.1)
    payment_effect = np.where(payment_method == "Electronic Check", 0.35, -0.05)
    ticket_effect = 0.12 * support_tickets

    logit = (
        base
        + contract_effect
        + tenure_effect
        + charge_effect
        + support_effect
        + payment_effect
        + ticket_effect
        + rng.normal(0, 0.6, size=n)  # noise
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, size=n) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(n)],
            "tenure_months": tenure_months,
            "contract": contract,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "tech_support": tech_support,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "support_tickets": support_tickets,
            "churn": churn,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_customers()
    df.to_csv("data/customers.csv", index=False)
    print(f"Generated {len(df)} synthetic customer records -> data/customers.csv")
    print(f"Overall churn rate: {df['churn'].mean():.1%}")

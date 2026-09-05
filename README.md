# Customer Churn Prediction — Telecom (Hypothetical Project)

Companion code repository for the **Week 4 Capstone: Comprehensive Data Science
Report and Insights Presentation Plan** (Yuva Intern / NSDC Virtual Data
Science Internship).

This project simulates an end-to-end churn-prediction workflow for a
subscription telecom provider: synthetic data generation → exploratory
analysis → model training and comparison → visualization of results. All
data is synthetically generated; no real customer data is used anywhere in
this repository.

## Project Structure

```
churn-project/
├── data/
│   └── customers.csv          # generated synthetic dataset (7,000 records)
├── src/
│   ├── generate_data.py       # creates the synthetic customer dataset
│   ├── eda.py                 # churn-by-contract and monthly trend charts
│   └── train_model.py         # trains & compares 3 classifiers, saves charts + metrics
├── outputs/
│   ├── figures/                # all chart PNGs referenced in the report
│   └── metrics.json            # accuracy / precision / recall + confusion matrix
├── requirements.txt
└── README.md
```

## What This Reproduces

This code generates the four visualizations and the confusion-matrix figures
used in the accompanying Word report:

1. **Churn rate by contract type** — bar chart
2. **Monthly churn trend** — line chart (illustrative seasonality overlay)
3. **Feature importance** — top predictors of churn from a Gradient Boosting model
4. **Model performance comparison** — Logistic Regression vs. Random Forest vs. Gradient Boosting

## How to Run

```bash
pip install -r requirements.txt

python src/generate_data.py    # -> data/customers.csv
python src/eda.py              # -> outputs/figures/chart_churn_by_contract.png, chart_churn_trend.png
python src/train_model.py      # -> outputs/figures/chart_model_comparison.png, chart_feature_importance.png
                                # -> outputs/metrics.json
```

## Methodology Summary

- **Data:** 7,000 synthetic customer records with tenure, contract type,
  monthly/total charges, tech support, internet service, payment method,
  and support-ticket history. Churn labels are generated from a logistic
  model with realistic effect sizes (e.g., month-to-month contracts and
  low tenure sharply increase churn probability), plus random noise.
- **Models compared:** Logistic Regression (baseline), Random Forest,
  Gradient Boosting — evaluated on accuracy, precision, and recall on a
  held-out 28% test split.
- **Why recall matters here:** missing an at-risk customer (false negative)
  is costlier to the business than a false alarm, so model selection
  favors recall alongside accuracy.

## Notes

- This is a hypothetical/educational project built for an internship
  capstone task — figures and conclusions illustrate the analytical
  approach and should not be interpreted as findings about any real
  company or dataset.
- Random seeds are fixed (`seed=42`) so results are reproducible.

## Author

Vrinda — Virtual Data Science Internship, Week 4 Capstone

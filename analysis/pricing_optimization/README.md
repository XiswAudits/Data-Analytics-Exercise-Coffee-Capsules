# Coffee Capsule Pricing Model

This folder is the legacy Streamlit entry point. `app.py` here changes to the repository root and runs the root application, so a host configured with `analysis/pricing_optimization/app.py` serves the same dashboard as `streamlit run app.py`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

From this directory the equivalent command is `streamlit run analysis/pricing_optimization/app.py`.

## Model

Revenue-maximising prices are a whiteboard linear programme in `price_optimisation.py`. Decision variables are the Regular price `R` and the Premium price `P`. Each household buys a fixed quantity `d_i` of one product. Household 1 and Household 3 use an OLS linear-probability line. Household 2 uses the horizontal threshold `P = 77.5` (Premium at or below it, Regular at or above it), because OLS misclassified week 6 (R = 55, P = 80). The page opens on revenue-maximising prices. A household control switches the chart between the shared programme and one household. Sensitivity, scenario comparison, and regret are computed from the same programme.

The formulation, fitted lines, and optima are documented in the repository [README](../../README.md).

## Dataset

The app reads `coffee_capsules_data.csv` at the repository root. `data/coffee_capsules.csv` records the same 11 weeks under this repository's earlier column names.

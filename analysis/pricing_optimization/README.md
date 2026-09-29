# Darbs Pricing Lab

Interactive Streamlit dashboard for household demand estimation, capsule price optimization, and scenario regret.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard includes an Apple-inspired light interface, household/capsule explorer, displayed fitted equations, interactive price response, revenue contour plot, and scenario payoff/regret views. All 11 weeks, including zero purchases, are retained.

**Model note:** revenue is nonlinear in prices when demand depends on price, so the dashboard uses bounded SLSQP optimization rather than a linear program. Own-price OLS slopes are capped at zero or below after fitting; this is a transparent shape adjustment, not constrained OLS.

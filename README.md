# Coffee Capsule Pricing Model

A reproducible pricing-analysis project using the 11-week, three-household dataset supplied for the exercise.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard includes Model Explanation, Demand Estimates, Feasible Region, Sensitivity, Scenarios & Regret, and Source Data.

## Dataset and method

`data/coffee_capsules.csv` transcribes all 11 weeks from the supplied table. Zero purchases are preserved. Six household/product demand equations are fitted with OLS using own price. If an estimated own-price slope is positive, the model uses a zero slope for that equation; both raw and adjusted slopes are reported.

Price bounds are derived from observed minimum and maximum prices. A linear price-ladder constraint requires Premium − Regular to be at least the smallest observed gap.

The original price × demand objective is nonlinear. The model samples product revenue curves on a fine grid and solves a piecewise-linear convex-hull approximation with `scipy.optimize.linprog`. This is an LP approximation, not an exact solution to the original nonlinear problem. Reported revenue is re-evaluated at the interpolated prices.

Sensitivity perturbs fitted demand intercepts by ±10%. Low/Base/High scenarios scale demand by −10%, 0%, and +10%. Strategies are ODS (optimized prices), BS1 (observed mean prices), BS2 (observed 25th percentile Regular / 75th percentile Premium), and BS3 (the reverse). Payoff tables compare revenues; regret is scenario-best revenue minus strategy revenue.

## Revenue vs. profit

No unit costs were supplied, so the objective is **revenue maximization**, not profit maximization. With unit costs, contribution profit can be modeled as (Regular price − Regular unit cost) × Regular demand + (Premium price − Premium unit cost) × Premium demand.

## Limitations

Only 11 weekly observations are available. OLS fits and scenarios are illustrative, sensitive to sparse price variation, and should not be treated as causal demand estimates or reliable forecasts. The intercept-at-zero interpretation is an extrapolation.

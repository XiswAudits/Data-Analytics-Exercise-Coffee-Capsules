# Coffee Capsule Pricing Model

A reproducible pricing-analysis project using the 11-week, three-household dataset supplied for the exercise.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens with Model Explanation, Demand Estimates, Feasible Region, Sensitivity, Scenarios & Regret, and Source Data tabs.

## Dataset

`data/coffee_capsules.csv` transcribes all 11 weeks from the supplied table. Columns include week, Regular/Premium prices, and each household's Regular/Premium quantity. Zero purchases are preserved.

## Method

- Six household/product demand equations are fitted with OLS: quantity = intercept + slope × own price.
- If an estimated own-price slope is positive, the model uses a zero slope for that equation to preserve a non-increasing demand curve. Both raw and adjusted slopes are reported.
- Price bounds come from observed minimum and maximum prices. A linear price-ladder constraint requires Premium − Regular to be at least the smallest observed gap.
- The original price × demand objective is nonlinear. The model samples product revenue curves on a fine grid and solves a piecewise-linear convex-hull approximation with `scipy.optimize.linprog`. This is an LP approximation; its lambda interpolation is not an exact global solution to the original nonlinear model. Reported revenue is re-evaluated at the interpolated prices.
- Sensitivity perturbs fitted demand intercepts by ±10%. Scenarios scale demand by −10%, 0%, and +10%.
- Strategies: ODS = optimized prices; BS1 = observed mean prices; BS2 = observed 25th percentile Regular / 75th percentile Premium; BS3 = observed 75th percentile Regular / 25th percentile Premium. All benchmarks are data-derived and adjusted to satisfy price bounds and the ladder constraint.
- Payoff tables compare strategy revenue across scenarios. Regret is scenario-best revenue minus the strategy's revenue; the summary includes maximum and average regret.

## Revenue vs. profit

No unit costs were supplied, so the objective is **revenue maximization**, not profit maximization. If unit costs are later added, contribution profit can be modeled as (Regular price − Regular unit cost) × Regular demand + (Premium price − Premium unit cost) × Premium demand.

## Limitations

Only 11 weekly observations are available. OLS fits and scenario results are illustrative, sensitive to sparse price variation, and should not be treated as causal demand estimates or reliable forecasts. The intercept-at-zero interpretation is an extrapolation. The LP approximation and shape adjustment are explicitly disclosed.

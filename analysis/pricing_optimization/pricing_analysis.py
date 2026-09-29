"""
Capsule pricing: household-level demand, revenue optimization, robustness, and regret.
Run: python pricing_analysis.py
Dependencies: pandas numpy scipy statsmodels matplotlib seaborn
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import statsmodels.api as sm

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def build_data():
    """Keep every week, including zero-purchase observations."""
    return pd.DataFrame({
        "Week": range(1, 12),
        "Regular_Price": [40,35,50,65,50,55,40,45,47,35,45],
        "Premium_Price": [70,81,100,75,70,80,90,92,91,70,80],
        "HH1_Regular": [5,5,4,0,0,4,4,4,4,5,4],
        "HH1_Premium": [0,0,0,4,4,0,0,0,0,0,0],
        "HH2_Regular": [0,4,4,0,0,4,4,4,4,0,4],
        "HH2_Premium": [3,0,0,3,3,0,0,0,0,3,0],
        "HH3_Regular": [0,0,0,0,0,0,0,0,0,0,0],
        "HH3_Premium": [4,4,0,4,5,4,0,0,0,5,4],
    })

def fit_household_models(df):
    """
    OLS per household/product: Q = intercept + b_R*RegularPrice + b_P*PremiumPrice.
    Economic monotonicity is imposed after OLS by projecting own-price coefficients
    to <= 0; cross-price coefficients remain unconstrained. This is a transparent
    shape restriction, not a constrained-OLS re-estimation.
    """
    X = sm.add_constant(df[["Regular_Price", "Premium_Price"]])
    models = {}
    for hh in ("HH1", "HH2", "HH3"):
        for product in ("Regular", "Premium"):
            y = df[f"{hh}_{product}"]
            result = sm.OLS(y, X).fit()
            b = result.params.copy()
            own = "Regular_Price" if product == "Regular" else "Premium_Price"
            b[own] = min(float(b[own]), 0.0)
            models[(hh, product)] = {
                "ols": result, "coef": b.to_numpy(dtype=float)
            }
    return models

def aggregate_demand(models, regular_price, premium_price, shocks=None):
    """Sum household forecasts; clip quantities at zero for economic feasibility."""
    x = np.array([1.0, regular_price, premium_price])
    totals = {"Regular": 0.0, "Premium": 0.0}
    for (hh, product), model in models.items():
        coef = model["coef"].copy()
        if shocks and (hh, product) in shocks:
            coef *= shocks[(hh, product)]
        q = max(0.0, float(x @ coef))
        totals[product] += q
    return totals

def revenue(models, r, p, shocks=None):
    q = aggregate_demand(models, r, p, shocks)
    return r*q["Regular"] + p*q["Premium"]

def optimize_prices(models, r_bounds=(20, 100), p_bounds=(30, 120), shocks=None):
    """Nonlinear SLSQP optimization because nonnegative demand clipping is piecewise."""
    objective = lambda z: -revenue(models, z[0], z[1], shocks)
    result = minimize(objective, x0=[np.mean(r_bounds), np.mean(p_bounds)],
                      method="SLSQP", bounds=[r_bounds, p_bounds],
                      options={"maxiter": 2000, "ftol": 1e-9})
    if not result.success:
        print("Optimizer warning:", result.message)
    return float(result.x[0]), float(result.x[1]), float(-result.fun)

def feasible_region_plot(models, optimum, r_bounds=(20,100), p_bounds=(30,120)):
    """Plot rectangular price bounds and iso-revenue contours over that region."""
    r = np.linspace(*r_bounds, 220)
    p = np.linspace(*p_bounds, 220)
    R, P = np.meshgrid(r, p)
    Z = np.empty_like(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            Z[i,j] = revenue(models, R[i,j], P[i,j])
    fig, ax = plt.subplots(figsize=(10,7))
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 12)
    cs = ax.contour(R, P, Z, levels=levels, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
    ax.axvline(r_bounds[0], color="gray", ls="--", label="Regular price bounds")
    ax.axvline(r_bounds[1], color="gray", ls="--")
    ax.axhline(p_bounds[0], color="gray", ls=":", label="Premium price bounds")
    ax.axhline(p_bounds[1], color="gray", ls=":")
    ax.scatter(optimum[0], optimum[1], s=130, marker="*", color="red",
               edgecolor="black", label="Optimal (R, P)", zorder=5)
    ax.set(xlabel="Regular price (R)", ylabel="Premium price (P)",
           title="Feasible price region and iso-revenue contours")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/"feasible_region.png", dpi=180)
    plt.show()

def perturbation_analysis(models, base_opt, pct=0.10, seed=7, n=200):
    """
    Multiplicatively perturb each fitted coefficient independently within +/-pct.
    Intercepts are perturbed too; zero coefficients remain zero.
    """
    rng = np.random.default_rng(seed)
    records = []
    for k in range(n):
        shocks = {key: rng.uniform(1-pct, 1+pct) for key in models}
        r, p, rev = optimize_prices(models, shocks=shocks)
        records.append({"run": k+1, "Regular_opt": r, "Premium_opt": p,
                        "Revenue_opt": rev})
    result = pd.DataFrame(records)
    result.to_csv(OUT/"sensitivity_runs.csv", index=False)
    print("\nPerturbation summary (+/-10% coefficient scaling):")
    print(result[["Regular_opt","Premium_opt","Revenue_opt"]].describe().round(2))
    return result

def scenario_and_regret(models, optimum):
    """Compare ODS and two bounded strategies under low/base/high coefficient states."""
    strategies = {
        "ODS": (optimum[0], optimum[1]),
        "BS1": (60.0, 80.0),   # (Regular, Premium)
        "BS2": (40.0, 100.0),
    }
    states = {"Low": 0.90, "Base": 1.00, "High": 1.10}
    payoff = pd.DataFrame(index=strategies, columns=states, dtype=float)
    for strategy, (r,p) in strategies.items():
        for state, scale in states.items():
            # Uniform scaling represents a simple demand-level scenario.
            shocks = {key: scale for key in models}
            payoff.loc[strategy,state] = revenue(models,r,p,shocks)
    regret = payoff.copy()
    for state in states:
        regret[state] = payoff[state].max() - payoff[state]
    minimax = regret.max(axis=1)
    print("\nPayoff table (revenue):")
    print(payoff.round(2))
    print("\nRegret matrix:")
    print(regret.round(2))
    print("\nMaximum regret by strategy:")
    print(minimax.round(2))
    print("Minimax-regret strategy:", minimax.idxmin())
    payoff.to_csv(OUT/"payoff_table.csv")
    regret.to_csv(OUT/"regret_matrix.csv")
    return payoff, regret

def main():
    df = build_data()
    # Averages are computed over all 11 weeks; zeros are intentionally retained.
    household_averages = df[[c for c in df if c.startswith("HH")]].mean()
    print("Average weekly demand (zeros included):")
    print(household_averages.round(3))
    models = fit_household_models(df)
    print("\nOLS coefficients (own-price slopes capped at <= 0 for optimization):")
    for key, model in models.items():
        print(key, dict(zip(["Intercept","b_Regular","b_Premium"], model["coef"].round(4))))
    r, p, rev = optimize_prices(models)
    print(f"\nOptimal Regular price R*: {r:.2f}")
    print(f"Optimal Premium price P*: {p:.2f}")
    print(f"Maximum modeled revenue: {rev:.2f}")
    feasible_region_plot(models, (r,p))
    perturbation_analysis(models, (r,p))
    scenario_and_regret(models, (r,p))

if __name__ == "__main__":
    main()

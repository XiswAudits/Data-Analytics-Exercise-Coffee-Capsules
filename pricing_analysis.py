"""Coffee capsule pricing model: OLS demand, LP approximation, scenarios and regret."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "coffee_capsules.csv"
HOUSEHOLDS = [1, 2, 3]
PRODUCTS = ["regular", "premium"]

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    required = ["week", "regular_price", "premium_price"] + [
        f"h{h}_{p}" for h in HOUSEHOLDS for p in PRODUCTS
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df

def fit_demand(df):
    """Fit household/product OLS on all weeks, including zero-purchase weeks."""
    models, rows = {}, []
    for h in HOUSEHOLDS:
        for product in PRODUCTS:
            x = df[f"{product}_price"]
            y = df[f"h{h}_{product}"]
            fit = sm.OLS(y.astype(float), sm.add_constant(x.astype(float))).fit()
            # Impose a transparent downward-sloping own-price shape if OLS slope is positive.
            intercept = float(fit.params.iloc[0])
            raw_slope = float(fit.params.iloc[1])
            slope = min(raw_slope, 0.0)
            models[(h, product)] = (intercept, slope)
            rows.append({
                "Household": f"Household {h}", "Product": product.title(),
                "Intercept": intercept, "OLS slope": raw_slope,
                "Model slope (non-positive)": slope, "R-squared": float(fit.rsquared),
                "Observations": int(fit.nobs)
            })
    return models, pd.DataFrame(rows)

def demand_at(models, product, price, scale=1.0, parameter_shift=0.0):
    """Aggregate household demand; scale is scenario demand shift."""
    q = 0.0
    for h in HOUSEHOLDS:
        a, b = models[(h, product)]
        # Simple sensitivity perturbs fitted intercept by +/-10%; slope remains fitted.
        a = a * (1.0 + parameter_shift)
        q += max(0.0, a + b * float(price))
    return max(0.0, q * scale)

def price_bounds(df):
    return {
        "regular": (float(df.regular_price.min()), float(df.regular_price.max())),
        "premium": (float(df.premium_price.min()), float(df.premium_price.max()))
    }

def optimize_lp(df, models, demand_scale=1.0, intercept_shift=0.0, grid_size=81):
    """
    LP convex-hull/piecewise-linear approximation.
    Each product's revenue curve is sampled at a data-bounded price grid.
    Lambda weights interpolate price and revenue; a premium >= regular + observed
    minimum premium-price gap keeps the product ladder feasible.
    """
    bounds = price_bounds(df)
    rlo, rhi = bounds["regular"]
    plo, phi = bounds["premium"]
    gap = max(0.0, float((df.premium_price - df.regular_price).min()))
    rgrid = np.linspace(rlo, rhi, grid_size)
    pgrid = np.linspace(plo, phi, grid_size)
    rq = np.array([demand_at(models, "regular", x, demand_scale, intercept_shift) for x in rgrid])
    pq = np.array([demand_at(models, "premium", x, demand_scale, intercept_shift) for x in pgrid])
    rrev, prev = rgrid * rq, pgrid * pq
    n = grid_size
    # Variables: regular lambdas [0:n], premium lambdas [n:2n]
    c = -np.r_[rrev, prev]
    A_eq = np.zeros((2, 2*n)); A_eq[0, :n] = 1; A_eq[1, n:] = 1
    b_eq = np.ones(2)
    # premium price - regular price >= gap
    A_ub = np.zeros((1, 2*n))
    A_ub[0, :n] = rgrid
    A_ub[0, n:] = -pgrid
    b_ub = np.array([-gap])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"LP optimization failed: {res.message}")
    lr, lp = res.x[:n], res.x[n:]
    rp = float(lr @ rgrid); pp = float(lp @ pgrid)
    # Evaluate fitted demand directly at reported prices (not just interpolated objective).
    q_r = demand_at(models, "regular", rp, demand_scale, intercept_shift)
    q_p = demand_at(models, "premium", pp, demand_scale, intercept_shift)
    return {"regular_price": rp, "premium_price": pp, "regular_demand": q_r,
            "premium_demand": q_p, "revenue": rp*q_r + pp*q_p,
            "lp_interpolated_revenue": float(-res.fun), "price_gap": gap,
            "success": bool(res.success)}

def make_benchmarks(df, optimized):
    # Price choices are derived from the observed price distribution, not hardcoded.
    rlo, rhi = df.regular_price.min(), df.regular_price.max()
    plo, phi = df.premium_price.min(), df.premium_price.max()
    rmean, pmean = df.regular_price.mean(), df.premium_price.mean()
    gap = max(0.0, float((df.premium_price-df.regular_price).min()))
    def feasible(r,p):
        return float(r), float(max(p, r+gap))
    bs2 = feasible(df.regular_price.quantile(.25), df.premium_price.quantile(.75))
    bs3 = feasible(df.regular_price.quantile(.75), df.premium_price.quantile(.25))
    # Clip to observed bounds while retaining the price-ladder constraint.
    def clip_pair(pair):
        r = float(np.clip(pair[0], rlo, rhi))
        p = float(np.clip(pair[1], plo, phi))
        if p < r+gap:
            p = min(phi, r+gap)
            if p < r+gap:
                r = max(rlo, p-gap)
        return r,p
    return {
        "ODS (LP optimized)": clip_pair((optimized["regular_price"], optimized["premium_price"])),
        "BS1 (observed mean prices)": clip_pair((rmean,pmean)),
        "BS2 (lower Regular / higher Premium)": clip_pair(bs2),
        "BS3 (higher Regular / lower Premium)": clip_pair(bs3),
    }

def scenario_tables(df, models, strategies):
    scenarios = {"Low demand (-10%)": .9, "Base demand": 1.0, "High demand (+10%)": 1.1}
    rows=[]
    for scenario, scale in scenarios.items():
        for strategy, (r,p) in strategies.items():
            qr=demand_at(models,"regular",r,scale)
            qp=demand_at(models,"premium",p,scale)
            rows.append({"Scenario":scenario,"Strategy":strategy,
                         "Regular price":r,"Premium price":p,
                         "Regular demand":qr,"Premium demand":qp,
                         "Revenue":r*qr+p*qp})
    payoff=pd.DataFrame(rows)
    matrix=payoff.pivot(index="Strategy",columns="Scenario",values="Revenue")
    regret=matrix.max(axis=0)-matrix
    summary=pd.DataFrame({"Maximum regret":regret.max(axis=1),
                          "Average regret":regret.mean(axis=1)})
    return payoff,matrix,regret,summary

def sensitivity(df, models, baseline):
    rows=[]
    for label, shift in [("-10% intercept",-.10),("Baseline",0.0),("+10% intercept",.10)]:
        result=optimize_lp(df,models,intercept_shift=shift)
        rows.append({"Demand parameter case":label,"Regular price":result["regular_price"],
                     "Premium price":result["premium_price"],"Revenue":result["revenue"],
                     "Revenue change vs baseline":result["revenue"]-baseline["revenue"]})
    return pd.DataFrame(rows)

def run_analysis(path=DATA_PATH):
    df=load_data(path)
    models, estimates=fit_demand(df)
    baseline=optimize_lp(df,models)
    strategies=make_benchmarks(df,baseline)
    payoff,matrix,regret,regret_summary=scenario_tables(df,models,strategies)
    sens=sensitivity(df,models,baseline)
    return {"data":df,"models":models,"estimates":estimates,"baseline":baseline,
            "strategies":strategies,"payoff":payoff,"matrix":matrix,
            "regret":regret,"regret_summary":regret_summary,"sensitivity":sens}

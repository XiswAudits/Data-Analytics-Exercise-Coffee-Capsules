import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pricing_analysis import run_analysis, demand_at, optimize_lp, price_bounds

st.set_page_config(page_title="Coffee Capsule Pricing Model", page_icon="☕", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
h1,h2,h3 {letter-spacing:-.025em;}
div[data-testid="stMetric"] {background:#f6f7f8;padding:16px;border-radius:12px;border:1px solid #e7e7e7;}
</style>
""", unsafe_allow_html=True)
st.title("Coffee Capsule Pricing Model")
st.caption("Household demand estimation · Price optimization · Scenario analysis")
try:
    out=run_analysis()
except Exception as e:
    st.error(f"Could not run model: {e}")
    st.stop()
df=out["data"]; opt=out["baseline"]; models=out["models"]; bounds=price_bounds(df)
st.info("Objective: maximize estimated revenue. Unit costs were not provided, so this is not profit maximization. Demand is estimated separately for each household and product using all 11 weeks, including zero-purchase observations.")
a,b,c=st.columns(3)
a.metric("Optimized Regular price",f"{opt['regular_price']:.2f}")
b.metric("Optimized Premium price",f"{opt['premium_price']:.2f}")
c.metric("Estimated weekly revenue",f"{opt['revenue']:.2f}")
st.caption(f"Observed price bounds: Regular {bounds['regular'][0]:.0f}–{bounds['regular'][1]:.0f}; Premium {bounds['premium'][0]:.0f}–{bounds['premium'][1]:.0f}. Product ladder constraint: Premium − Regular ≥ {opt['price_gap']:.0f} (minimum observed price gap).")
tabs=st.tabs(["Model Explanation","Demand Estimates","Feasible Region","Sensitivity","Scenarios & Regret","Data"])
with tabs[0]:
    st.header("Model Explanation")
    st.markdown("""
**1. Demand estimation.** For each of the three households and each capsule type, ordinary least squares estimates quantity as a function of that product's price. The zero-purchase weeks are retained, so the estimates reflect both buying and not buying in the observed period. A positive fitted own-price slope is reset to zero to preserve a non-increasing demand response; the raw and adjusted slopes are both shown.

**2. Price bounds and feasibility.** Prices are restricted to the minimum and maximum values observed in the data. The model also keeps Premium at least as expensive as Regular by the smallest premium–regular price difference observed in the 11 weeks.

**3. Optimization.** Revenue is Regular price × aggregate Regular demand + Premium price × aggregate Premium demand. Since price multiplied by price-dependent demand is nonlinear, the model samples revenue over a fine price grid and solves a linear-programming convex-hull interpolation with `scipy.optimize.linprog`. This is an LP approximation, not an exact solution to the original nonlinear problem. Reported revenue is re-evaluated from the fitted demand equations at the interpolated prices.

**4. Sensitivity and strategy comparison.** Sensitivity changes each fitted intercept by ±10%. Low/Base/High scenarios scale demand by −10%/0%/+10%. ODS is the LP price pair; BS1 uses observed mean prices; BS2 uses the observed 25th percentile Regular and 75th percentile Premium; BS3 reverses those quantiles. The payoff table reports revenue; regret is the difference from the best strategy within each scenario.
""")
with tabs[1]:
    st.header("Demand Estimates")
    st.write("Each row is a household/product OLS fit. R-squared is descriptive and should be interpreted cautiously with only 11 observations.")
    st.dataframe(out["estimates"].round(4),use_container_width=True,hide_index=True)
    st.markdown("**How to read it:** the intercept is the fitted quantity at price zero (an extrapolation, not a practical forecast); the adjusted slope is the model's own-price response.")
with tabs[2]:
    st.header("Feasible Region")
    st.write("The shaded area shows prices within observed ranges and satisfying the Premium price ladder.")
    rlo,rhi=bounds["regular"]; plo,phi=bounds["premium"]
    import numpy as np
    xs=np.linspace(rlo,rhi,200); ys=np.linspace(plo,phi,200)
    X,Y=np.meshgrid(xs,ys)
    feasible=(Y-X>=opt["price_gap"])
    fig,ax=plt.subplots(figsize=(8,5))
    ax.contourf(X,Y,feasible,levels=[.5,1.5],alpha=.25)
    ax.plot(xs,xs+opt["price_gap"],label="Premium = Regular + minimum observed gap")
    ax.scatter(df.regular_price,df.premium_price,label="Observed weekly prices",s=30)
    ax.scatter([opt["regular_price"]],[opt["premium_price"]],marker="*",s=180,label="LP solution")
    ax.set(xlabel="Regular price",ylabel="Premium price",xlim=(rlo,rhi),ylim=(plo,phi))
    ax.legend(); ax.grid(alpha=.2); st.pyplot(fig); plt.close(fig)
    st.caption("The region is generated from observed price bounds and the stated linear ladder constraint; no price values are manually selected.")
with tabs[3]:
    st.header("Sensitivity Analysis")
    st.write("The intercepts of the fitted household demand equations are perturbed together by −10% and +10%, and the LP is re-solved.")
    st.dataframe(out["sensitivity"].round(3),use_container_width=True,hide_index=True)
    st.caption("Price changes indicate how sensitive the recommended price pair is to this simple demand-level perturbation. This is not a statistical confidence interval.")
with tabs[4]:
    st.header("Scenarios, Payoff & Regret")
    st.subheader("Revenue payoff table")
    st.dataframe(out["matrix"].round(2),use_container_width=True)
    st.subheader("Regret / opportunity loss")
    st.write("For each scenario, regret = best revenue available among the four strategies − revenue of the strategy in that scenario.")
    st.dataframe(out["regret"].round(2),use_container_width=True)
    st.subheader("Regret summary")
    st.dataframe(out["regret_summary"].round(2),use_container_width=True)
    st.caption("The maximum-regret column supports a minimax-regret decision rule; it is a summary criterion, not a forecast.")
with tabs[5]:
    st.header("Source Data")
    st.write("Transcribed from the 11-week table supplied for this exercise.")
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.download_button("Download dataset (CSV)",df.to_csv(index=False).encode("utf-8"),"coffee_capsules.csv","text/csv")

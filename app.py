"""Streamlit page for the whiteboard price linear programme.

The logistic Buy / No-Purchase classifier is not part of this page. Prices,
quantities, lines, and optima all come from `price_optimisation.py`.
"""

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from price_optimisation import (
    bound_sensitivity,
    bound_sensitivity_figure,
    constraint_table,
    household_equations,
    household_figure,
    optimise_prices_detailed,
    price_move_figure,
    regret_figure,
    regret_table,
    revenue_of,
    shared_figure,
    structural_sensitivity,
    structural_sensitivity_figure,
)

DATA_CSV = str(Path(__file__).resolve().parent / "coffee_capsules_data.csv")
ALL_VIEW = "All households (shared prices)"

st.set_page_config(page_title="Revenue-maximising prices", page_icon="☕", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;}
.stApp, [data-testid="stAppViewContainer"] {background: #f3f5f7; color: #101828;}
[data-testid="stSidebar"] {display: none;}
.block-container {max-width: 1180px; padding: 28px 28px 64px;}
h1 {font-size: 36px; line-height: 1.08; letter-spacing: -1.4px; font-weight: 600; color: #101828; margin: 0 0 8px;}
.lede {color: #667085; font-size: 15px; line-height: 1.55; max-width: 720px; margin: 0 0 22px;}
.eyebrow {display: inline-block; padding: 6px 10px; border-radius: 999px; background: #edf3f9; color: #4b6380; font-size: 11px; font-weight: 600; margin-bottom: 10px;}
.card {background: #fff; border: 1px solid #e5eaf0; border-radius: 18px; padding: 16px 18px; box-shadow: 0 5px 20px rgba(16,24,40,.035); height: 100%;}
.card-title {color: #101828; font-size: 14px; font-weight: 600; margin-bottom: 4px;}
.card-sub {color: #98a2b3; font-size: 12px; line-height: 1.5;}
.equation {background: #f8fafc; border: 1px solid #e7edf3; border-radius: 12px; padding: 12px 14px; color: #172033; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.45;}
.note {color: #667085; font-size: 12px; line-height: 1.5; margin: 8px 0 0;}
.insight {background: linear-gradient(160deg, #172033, #24384f); color: #fff; border-radius: 18px; padding: 22px 20px; min-height: 100%; box-shadow: 0 14px 30px rgba(23,32,51,.16);}
.insight .eyebrow {background: rgba(255,255,255,.12); color: #dbeafe;}
.insight h2 {color: white; font-size: 28px; letter-spacing: -0.8px; margin: 6px 0 4px; font-weight: 600;}
.insight .sub {color: #cbd5e1; font-size: 13px; margin: 0 0 12px;}
.prob-row {display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,.1); font-size: 13px;}
.prob-row:last-child {border-bottom: 0;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #e5eaf0; border-radius: 16px; padding: 14px 16px; box-shadow: 0 4px 16px rgba(16,24,40,.035);}
[data-testid="stMetricLabel"] {font-size: 12px !important; color: #667085 !important;}
[data-testid="stMetricValue"] {font-size: 26px !important; color: #101828 !important; letter-spacing: -0.6px;}
div[data-testid="stSegmentedControl"] {margin: 4px 0 8px;}
[data-testid="stDataFrame"] {border-radius: 14px; overflow: hidden;}
@media (max-width: 800px) {
  .block-container {padding: 16px 12px 40px;}
  h1 {font-size: 28px;}
  [data-testid="stMetricValue"] {font-size: 22px !important;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def household_price_lp(csv_name: str):
    """Shared and per-household programmes. Cached so the view switch does not refit."""
    detail = optimise_prices_detailed(csv_name)
    structural = structural_sensitivity(detail, pct=0.10)
    bounds = bound_sensitivity(detail)
    return detail, structural, bounds


def money(value: float) -> str:
    return f"€{value:,.2f}"


def short_assignment(assignment: dict) -> str:
    return " · ".join(f"H{label.split()[-1]} {product}" for label, product in assignment.items())


def active_programme(detail: dict, view: str) -> tuple[dict, list[dict], str]:
    """Solved programme, the households it is about, and a chart title."""
    if view == ALL_VIEW:
        return detail["shared"]["best"], detail["households"], "Shared prices"
    item = next(entry for entry in detail["per_household"] if entry["household"]["label"] == view)
    return item["programme"]["best"], [item["household"]], view


def demand_table(households: list[dict]) -> pd.DataFrame:
    rows = []
    for household in households:
        rows.append({
            "Household": household["label"],
            "d Regular": household["d_regular"],
            "Regular weeks": household["n_regular"],
            "d Premium": household["d_premium"],
            "Premium weeks": household["n_premium"],
        })
    return pd.DataFrame(rows)


def demand_figure(households: list[dict]) -> go.Figure:
    labels = [household["label"] for household in households]
    fig = go.Figure()
    fig.add_bar(name="Regular", x=labels, y=[household["d_regular"] for household in households], marker_color="#1677ff")
    fig.add_bar(name="Premium", x=labels, y=[household["d_premium"] for household in households], marker_color="#ff8a1f")
    fig.update_layout(
        barmode="group",
        height=320,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=48, r=16, t=36, b=40),
        legend=dict(orientation="h", y=1.14),
        yaxis_title="Capsules on weeks they bought it",
        yaxis=dict(gridcolor="#e8edf3", zeroline=False),
        xaxis=dict(zeroline=False),
        font=dict(family="Inter, Arial, sans-serif", color="#172033"),
    )
    return fig


def result_card(solved: dict, households: list[dict], view: str) -> str:
    product_rows = []
    for household in households:
        product = solved["assignment"][household["label"]]
        quantity = {"Regular": household["d_regular"], "Premium": household["d_premium"], "None": 0.0}[product]
        product_rows.append(
            f'<div class="prob-row"><span>{household["label"]}</span><b>{product} · d = {quantity:.2f}</b></div>'
        )
    title = "Shared menu" if view == ALL_VIEW else solved["assignment"][view]
    subtitle = "All households, one price pair" if view == ALL_VIEW else f"{view} solved on its own line"
    tight = solved.get("tight") or []
    if tight:
        binding_rows = "".join(
            f'<div class="prob-row"><span>Binding</span><b>{escape(name.split(" so that ", 1)[-1])}</b></div>'
            for name in tight
        )
    else:
        binding_rows = '<div class="prob-row"><span>Binding</span><b>none</b></div>'
    return (
        '<div class="insight">'
        '<div class="eyebrow">Optimum</div>'
        f"<h2>{escape(str(title))}</h2>"
        f'<p class="sub">{escape(subtitle)}</p>'
        f'<div class="prob-row"><span>Regular price R</span><b>{money(solved["R"])}</b></div>'
        f'<div class="prob-row"><span>Premium price P</span><b>{money(solved["P"])}</b></div>'
        f'<div class="prob-row"><span>Weekly revenue</span><b>{money(solved["revenue"])}</b></div>'
        + "".join(product_rows)
        + binding_rows
        + "</div>"
    )


detail, structural, bound_table = household_price_lp(DATA_CSV)
shared = detail["shared"]["best"]
labels = [ALL_VIEW] + [item["household"]["label"] for item in detail["per_household"]]

st.markdown(
    '<div class="eyebrow">Whiteboard linear programme</div>'
    "<h1>Revenue-maximising prices</h1>"
    '<p class="lede">Each household buys a fixed quantity of one product, so revenue is linear in the Regular price R and the Premium price P. '
    "A line in that plane decides the product. The shared menu is the feasible assignment with the highest revenue.</p>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Shared Regular price", money(shared["R"]))
k2.metric("Shared Premium price", money(shared["P"]))
k3.metric("Total weekly revenue", money(shared["revenue"]))
k4.metric("Assignment", short_assignment(shared["assignment"]))

st.markdown('<div class="card-title" style="margin-top:18px">Household</div>', unsafe_allow_html=True)
view = st.segmented_control(
    "Household",
    options=labels,
    default=ALL_VIEW,
    key="lp_view",
    label_visibility="collapsed",
    width="stretch",
)
if view is None:
    view = ALL_VIEW

solved, shown, chart_title = active_programme(detail, view)
figure = shared_figure(detail) if view == ALL_VIEW else household_figure(detail, view)
chart_note = (
    "Every household line, the shaded feasible region, and the shared optimum."
    if view == ALL_VIEW
    else "This household's line, the weeks they were observed, the feasible region, and their own optimum."
)

left, right = st.columns([1.65, 0.85], gap="medium")
with left:
    st.markdown(
        f'<div class="card"><div class="card-title">{chart_title}</div><div class="card-sub">{chart_note}</div></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False, "responsive": True})
with right:
    st.markdown(result_card(solved, shown, view), unsafe_allow_html=True)
    if solved.get("flat_price"):
        st.caption(solved["flat_price"])

formula_cols = st.columns(1 + len(shown))
with formula_cols[0]:
    objective = escape(constraint_table(solved, shown).iloc[0]["Constraint"])
    st.markdown(f'<div class="equation">Objective<br>{objective}</div>', unsafe_allow_html=True)
for column, household in zip(formula_cols[1:], shown):
    with column:
        lines = "<br>".join(escape(line) for line in household_equations(household))
        st.markdown(f'<div class="equation">{escape(household["label"])}<br>{lines}</div>', unsafe_allow_html=True)

week6 = detail["frame"].loc[detail["frame"]["T"] == 6]
if not week6.empty:
    row = week6.iloc[0]
    st.markdown(
        f'<p class="note">Household 2&apos;s line is set to the separating threshold because OLS misclassified week 6 '
        f'(R = {row["P_Regular"]:.0f}, P = {row["P_Premium"]:.0f}).</p>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="card"><div class="card-title">Constraints</div>'
    '<div class="card-sub">Every side of each switching line, the price box, and the objective of the programme in view. '
    "Binding is evaluated at that programme's optimum. A side marked “not in this programme” is the other side of the line.</div></div>",
    unsafe_allow_html=True,
)
constraints = constraint_table(solved, detail["households"] if view == ALL_VIEW else shown)
st.dataframe(constraints, width="stretch", hide_index=True)

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="card"><div class="card-title">Average weekly demand</div>'
    '<div class="card-sub">d is the average number of capsules on weeks when that household bought the product. Weeks with a zero stay in the sample and are not part of this average.</div></div>',
    unsafe_allow_html=True,
)
demand_left, demand_right = st.columns([1, 1.15], gap="medium")
with demand_left:
    demand = demand_table(detail["households"])
    styled = demand.copy()
    styled["d Regular"] = styled["d Regular"].map(lambda value: f"{value:.4f}")
    styled["d Premium"] = styled["d Premium"].map(lambda value: f"{value:.4f}")
    st.dataframe(styled, width="stretch", hide_index=True)
with demand_right:
    st.plotly_chart(demand_figure(detail["households"]), width="stretch", config={"displayModeBar": False, "responsive": True})

line_shocks = structural[structural["kind"] == "line coefficient"]
price_moves = structural[structural["kind"] == "price move"]
if view != ALL_VIEW:
    line_shocks = line_shocks[line_shocks["Household"] == view]
    price_moves = price_moves[price_moves["Household"] == view]
if view == ALL_VIEW:
    shock_note = "Each OLS coefficient, and Household 2's threshold, is moved by 10 percent and the shared programme is solved again."
else:
    shock_note = f"{view}'s line is moved by 10 percent and the shared programme is solved again. Only that household is shown."

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="card"><div class="card-title">Sensitivity, ±10%</div>'
    f'<div class="card-sub">{escape(shock_note)} '
    "The next chart moves the household's own optimal price by 10 percent and recomputes revenue from the same lines.</div></div>",
    unsafe_allow_html=True,
)
if not line_shocks.empty:
    st.plotly_chart(structural_sensitivity_figure(line_shocks), width="stretch", config={"displayModeBar": False, "responsive": True})
    show = line_shocks[["Household", "parameter", "shock", "R_opt", "P_opt", "max_revenue", "delta_revenue", "assignment", "status"]].copy()
    show.columns = ["Household", "Term", "Shock", "R", "P", "Revenue", "Δ revenue", "Assignment", "Status"]
    st.dataframe(show.round(2), width="stretch", hide_index=True)
if not price_moves.empty:
    st.plotly_chart(price_move_figure(price_moves), width="stretch", config={"displayModeBar": False, "responsive": True})
if view == ALL_VIEW:
    st.plotly_chart(bound_sensitivity_figure(bound_table), width="stretch", config={"displayModeBar": False, "responsive": True})
    st.dataframe(bound_table.round(2), width="stretch", hide_index=True)

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="card"><div class="card-title">Scenario comparison and regret</div>'
    '<div class="card-sub">The scenario is a Regular / Premium price pair. Regret is a household&apos;s own LP revenue minus revenue at that menu. '
    "The cell on a household's own optimum is zero.</div></div>",
    unsafe_allow_html=True,
)
mean_r = round(float(detail["frame"]["P_Regular"].mean()), 2)
mean_p = round(float(detail["frame"]["P_Premium"].mean()), 2)
s1, s2 = st.columns(2)
scenario_r = s1.number_input(
    "Scenario Regular price",
    min_value=float(detail["bounds"]["R_lower"]),
    max_value=float(detail["bounds"]["R_upper"]),
    value=mean_r,
    step=0.01,
    key="scenario_r",
)
scenario_p = s2.number_input(
    "Scenario Premium price",
    min_value=float(detail["bounds"]["P_lower"]),
    max_value=float(detail["bounds"]["P_upper"]),
    value=mean_p,
    step=0.01,
    key="scenario_p",
)
regret = regret_table(detail, scenario=(scenario_r, scenario_p))
if view != ALL_VIEW:
    regret_view = regret[regret["Household"] == view]
else:
    regret_view = regret
st.plotly_chart(regret_figure(regret_view), width="stretch", config={"displayModeBar": False, "responsive": True})
rev_pivot = regret_view.pivot(index="Household", columns="Menu", values="revenue")
reg_pivot = regret_view.pivot(index="Household", columns="Menu", values="regret")
st.markdown("**Revenue by scenario (€)**")
st.dataframe(rev_pivot.round(2), width="stretch")
st.markdown("**Regret versus that household's LP optimum (€)**")
st.dataframe(reg_pivot.round(2), width="stretch")

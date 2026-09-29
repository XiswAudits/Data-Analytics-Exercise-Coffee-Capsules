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
    contour_terms,
    household_equations,
    household_figure,
    ols_switch_line,
    optimise_prices_detailed,
    predict_product,
    price_move_figure,
    regret_figure,
    regret_table,
    revenue_of,
    shared_figure,
    structural_sensitivity,
    structural_sensitivity_figure,
)

SHOCK_PCT = 0.10

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
    structural = structural_sensitivity(detail, pct=SHOCK_PCT)
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
shock_label = f"{SHOCK_PCT:.0%}"
if view == ALL_VIEW:
    shock_note = f"Each OLS coefficient, and Household 2's threshold, is moved by {shock_label} and the shared programme is solved again."
else:
    shock_note = f"{view}'s line is moved by {shock_label} and the shared programme is solved again. Only that household is shown."

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="card"><div class="card-title">Sensitivity, ±10%</div>'
    f'<div class="card-sub">{escape(shock_note)} '
    f"The next chart moves the household's own optimal price by {shock_label} and recomputes revenue from the same lines.</div></div>",
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


def _latex_line(slope: float, icept: float) -> str:
    if abs(slope) < 1e-8:
        return rf"P = {icept:.4f}"
    sign = "+" if icept >= 0 else "-"
    return rf"P = {slope:.4f}\, R {sign} {abs(icept):.4f}"


def _buying_span(frame: pd.DataFrame, household: dict, product: str) -> tuple[float, float] | None:
    column = f"{household['prefix']}_{product}"
    qty = frame[column].to_numpy(dtype=float)
    positive = qty[qty > 0]
    if len(positive) == 0:
        return None
    return float(positive.min()), float(positive.max())


def _ols_misses(frame: pd.DataFrame, household: dict) -> list[dict]:
    line = ols_switch_line(frame, household)
    if line is None:
        return []
    probe = {"lines": [line], "only_product": None}
    prefix = household["prefix"]
    misses = []
    for _, row in frame.iterrows():
        regular_qty = float(row[f"{prefix}_Regular"])
        premium_qty = float(row[f"{prefix}_Premium"])
        observed = "Regular" if regular_qty > 0 else "Premium" if premium_qty > 0 else "None"
        if observed == "None":
            continue
        predicted = predict_product(probe, float(row["P_Regular"]), float(row["P_Premium"]))
        if predicted != observed:
            misses.append({
                "week": int(row["T"]),
                "R": float(row["P_Regular"]),
                "P": float(row["P_Premium"]),
                "observed": observed,
                "predicted": predicted,
            })
    return misses


def render_methodology(detail: dict, bound_rows: pd.DataFrame) -> None:
    """End-to-end account of the linear programme. Every figure comes from `detail`."""
    frame = detail["frame"]
    households = detail["households"]
    bounds = detail["bounds"]
    shared = detail["shared"]["best"]
    cases = detail["shared"]["cases"]
    n_weeks = int(frame["T"].nunique())
    n_households = len(households)
    n_feasible = sum(1 for case in cases if case["feasible"])

    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card"><div class="card-title">Model & methodology</div>'
        '<div class="card-sub">How the prices on this page were obtained, from the weekly table to the optimum.</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### The data")
    st.markdown(
        f"The table has **{n_weeks} weekly observations** of three things: the Regular price R, the Premium price P, "
        f"and how many capsules each of the **{n_households} households** bought of each product. "
        "A week with a zero is kept. It is a real choice, not a missing row."
    )

    st.markdown("#### Fixed quantities")
    st.markdown(
        "On the weeks a household actually buys a product, the number of capsules barely moves. "
        "We freeze that number at its average, written d. Revenue is then just that quantity times a price, which is linear, so the programme below really is a linear programme."
    )
    st.latex(r"d_{i,\mathrm{product}} = \text{average capsules on weeks household } i \text{ bought that product}")
    st.latex(r"\mathrm{revenue} = \sum_i d_i \times (\text{price of the product household } i \text{ buys})")
    for household in households:
        bits = []
        for product, key in (("Regular", "d_regular"), ("Premium", "d_premium")):
            span = _buying_span(frame, household, product)
            if span is None:
                bits.append(f"never buys {product}")
                continue
            lo, hi = span
            spread = "the same number every buying week" if abs(hi - lo) < 1e-9 else f"only between {lo:.0f} and {hi:.0f}"
            bits.append(f"{product} d = {household[key]:.4f} ({spread})")
        st.markdown(f"- **{household['label']}:** " + "; ".join(bits) + ".")

    st.markdown("#### Switching lines")
    st.markdown(
        "Which product they buy is a straight line in the R–P plane. For a household that buys both, ordinary least squares fits a linear probability model: a 0/1 label (Premium = 1, Regular = 0) on an intercept, R, and P. The line drawn on the chart is the contour where that fitted score is one half."
    )
    st.latex(r"\mathrm{score} = a + b_R R + b_P P")
    st.latex(r"a + b_R R + b_P P = 0.5 \quad \Rightarrow \quad P = m R + k")
    for household in households:
        used = household["lines"]
        accuracy = _accuracy_label_local(household)
        if any(line.get("fixed_threshold") is not None for line in used):
            level = next(float(line["fixed_threshold"]) for line in used if line.get("fixed_threshold") is not None)
            ols = ols_switch_line(frame, household)
            misses = _ols_misses(frame, household)
            st.markdown(f"**{household['label']}.** The OLS contour on their buying weeks would be")
            if ols is not None and contour_terms(ols) is not None:
                st.latex(_latex_line(*contour_terms(ols)))
            if misses:
                described = "; ".join(
                    f"week {miss['week']} (R = {miss['R']:.0f}, P = {miss['P']:.0f}) was {miss['observed']}, "
                    f"while the OLS line called it {miss['predicted']}"
                    for miss in misses
                )
                st.markdown(f"It misclassifies {described}.")
            else:
                st.markdown("That fit matches every buying week.")
            st.markdown(
                f"The programme therefore uses the horizontal threshold that separates those weeks. "
                f"In-sample accuracy of the line actually used: {accuracy}."
            )
            st.latex(rf"P = {level:.4f}")
            st.markdown(
                f"Premium when P is at or below {level:.4f}, Regular when P is at or above {level:.4f}."
            )
            continue
        for line in used:
            terms = contour_terms(line)
            if terms is None:
                st.markdown(f"**{household['label']}:** {line['equation']} (in-sample accuracy {accuracy}).")
                continue
            st.markdown(f"**{household['label']}** — in-sample accuracy {accuracy}.")
            st.latex(_latex_line(*terms))
            if line["kind"] == "stop":
                st.markdown("They never buy Regular. This line is the contour between buying Premium and buying nothing.")
            else:
                st.markdown("Premium on the side where the score is at least one half, Regular on the other side.")

    st.markdown("#### The linear programme")
    st.markdown(
        f"Each way of assigning a product to the households is its own linear programme. "
        f"There are **{len(cases)}** such assignments. "
        f"`scipy.optimize.linprog` (HiGHS) solves every one; the solver minimises, so the objective vector is the negated quantities. "
        f"**{n_feasible}** assignments are feasible inside the price box. We keep the one with the highest revenue."
    )
    c_r, c_p = float(shared["c"][0]), float(shared["c"][1])
    st.latex(rf"\max \quad {c_r:.4f}\, R + {c_p:.4f}\, P")
    st.markdown("Subject to the chosen side of each household's line, and to the price box. The upper bounds are the highest prices in the sample. Prices also stay non-negative.")
    st.latex(rf"{bounds['R_lower']:.0f} \le R \le {bounds['R_upper']:.4f}, \qquad {bounds['P_lower']:.0f} \le P \le {bounds['P_upper']:.4f}")
    for name in shared["constraint_names"]:
        if name.startswith("R ") or name.startswith("P "):
            continue
        pretty = name.split(" so that ", 1)[-1].replace("−", "-")
        pretty = pretty.replace("<=", r"\le").replace(">=", r"\ge").replace(" R ", r" \, R ")
        st.latex(pretty)

    st.markdown("#### The result")
    binding = ", ".join(name.split(" so that ", 1)[-1] for name in shared["tight"]) or "none"
    st.markdown(
        f"The best feasible menu is **{shared['assignment_label']}**. "
        f"The optimum is R = {money(shared['R'])}, P = {money(shared['P'])}, weekly revenue {money(shared['revenue'])}. "
        f"It is a corner of the feasible set. Binding there: {binding}."
    )

    st.markdown("#### Sensitivity, scenarios, and regret")
    st.markdown(
        f"- **Line shocks.** Each OLS coefficient, and Household 2's threshold, is moved by {SHOCK_PCT:.0%} and the shared programme is solved again. "
        "That shows whether the menu and the revenue depend on a fragile slope."
    )
    st.markdown(
        f"- **Price moves.** Each household's own optimal price is moved by {SHOCK_PCT:.0%}, and revenue is read off the same lines. "
        "That is the cost of missing their own optimum by a little."
    )
    box_names = ", ".join(str(name) for name in bound_rows["Price box"])
    st.markdown(
        f"- **Price box.** The shared programme is solved again as the upper bounds relax ({box_names}). "
        "If revenue keeps climbing, the optimum was leaning on the edge of the sample."
    )
    st.markdown(
        "- **Scenario.** A Regular / Premium pair — by default the average prices in the table — is scored with the same lines. "
        "Menus in the comparison are the shared optimum, each household's own optimum, and that scenario."
    )
    st.latex(r"\mathrm{regret}_i(\mathrm{menu}) = \mathrm{revenue}_i(\mathrm{own\ optimum}) - \mathrm{revenue}_i(\mathrm{menu})")
    st.markdown("Regret is zero on a household's own optimum, and positive when another menu earns them less.")

    st.markdown("#### Limitations")
    at_ceiling = abs(shared["R"] - bounds["R_upper"]) <= 1e-6
    if at_ceiling:
        st.markdown(
            f"- **R is at the top of the sample.** The optimum sets R to {money(bounds['R_upper'])}, "
            f"the highest Regular price observed. The programme is not allowed to go higher, so this is a corner of the box, not evidence that a still higher Regular price would fail."
        )
    else:
        st.markdown(
            f"- The optimum Regular price is {money(shared['R'])}, inside the box whose top is the highest observed Regular price, {money(bounds['R_upper'])}."
        )
    st.markdown(
        f"- **The sample is small:** {n_weeks} weeks and {n_households} households. The lines and the optimum are a reading of this table, not a forecast."
    )
    st.markdown(
        "- **Quantities are fixed.** Inside a region, selling at a higher price does not reduce d. That is why revenue is linear, and why a price on the edge of a region looks attractive. It is not a demand curve."
    )


def _accuracy_label_local(household: dict) -> str:
    n = household["n_regular"] + household["n_premium"] + household["n_none"]
    correct = int(round(household["accuracy"] * n))
    return f"{correct}/{n}"


render_methodology(detail, bound_table)

"""Streamlit page for the whiteboard price linear programme.

The logistic Buy / No-Purchase classifier is not part of this page. Prices,
quantities, lines, and optima all come from `price_optimisation.py`.
"""

from html import escape
from pathlib import Path
import re

import importlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Streamlit Cloud reruns this file but keeps the previous price_optimisation
# module in sys.modules. Reload so a hot reload cannot import stale names.
import price_optimisation
importlib.reload(price_optimisation)

from price_optimisation import (
    apply_chart_layout,
    best_programme,
    classify_accuracy,
    constraint_table,
    contour_line,
    contour_terms,
    format_display_equation,
    household_figure,
    ols_regression,
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
# Lines a classmate might draw through the gap between groups, written P = m R + k.
# They are a comparison, not a fit. Accuracy and the shared optimum are computed from them.
ALTERNATIVE_CONTOURS = {
    "Household 1": (0.6667, 40.00),
    "Household 2": (0.1667, 67.50),
    "Household 3": (-0.0502, 87.38),
}
EXAMPLE_WEEK = 4

DATA_CSV = str(Path(__file__).resolve().parent / "coffee_capsules_data.csv")
ALL_VIEW = "All households (shared prices)"

st.set_page_config(page_title="Revenue-maximising prices", page_icon="☕", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, [data-testid="stMarkdown"], [data-testid="stCaptionContainer"],
label, input, button, [data-testid="stHeading"] {
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
.stApp, [data-testid="stAppViewContainer"] {background: #f3f5f7; color: #101828;}
[data-testid="stSidebar"] {display: none;}
.block-container {max-width: 1180px; padding: 28px 28px 64px;}
h1 {font-size: 36px; line-height: 1.08; letter-spacing: -1.4px; font-weight: 600; color: #101828; margin: 0 0 8px;}
h4 {font-size: 18px !important; line-height: 1.3 !important; font-weight: 600 !important; color: #101828 !important; letter-spacing: -0.3px; margin: 22px 0 6px !important;}
.lede {color: #667085; font-size: 15px; line-height: 1.55; max-width: 760px; margin: 0 0 18px;}
.eyebrow {display: inline-block; padding: 6px 10px; border-radius: 999px; background: #edf3f9; color: #4b6380; font-size: 11px; font-weight: 600; letter-spacing: 0.02em; margin-bottom: 10px;}
.card, .section-head {background: #fff; border: 1px solid #e5eaf0; border-radius: 18px; padding: 16px 18px; box-shadow: 0 5px 20px rgba(16,24,40,.035);}
.section-head {margin: 22px 0 12px;}
.card-title {color: #101828; font-size: 16px; font-weight: 600; letter-spacing: -0.2px; margin: 0 0 4px;}
.card-sub {color: #667085; font-size: 13px; line-height: 1.5;}
.switch-label {color: #101828; font-size: 13px; font-weight: 600; margin: 16px 0 6px;}
.kpi-row, .formula-grid {display: grid; gap: 12px; align-items: stretch;}
.kpi-row {grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 4px 0 6px;}
.kpi {background: #fff; border: 1px solid #e5eaf0; border-radius: 16px; padding: 14px 16px 16px; box-shadow: 0 4px 16px rgba(16,24,40,.035); min-height: 96px; display: flex; flex-direction: column;}
.kpi-label {font-size: 12px; line-height: 1.35; color: #667085; font-weight: 500;}
.kpi-value {margin-top: auto; padding-top: 8px; font-size: 26px; line-height: 1.15; letter-spacing: -0.6px; font-weight: 600; color: #101828;}
.kpi-value.compact {font-size: 15px; line-height: 1.35; letter-spacing: -0.2px; font-weight: 600;}
.formula-grid {grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); margin-top: 14px;}
.formula-card {background: #fff; border: 1px solid #e5eaf0; border-radius: 16px; padding: 14px 16px 16px; box-shadow: 0 5px 20px rgba(16,24,40,.035); min-height: 188px; height: 100%; display: flex; flex-direction: column;}
.formula-kicker {font-size: 12px; font-weight: 600; color: #4b6380; letter-spacing: 0.01em; margin-bottom: 8px;}
.formula-lead {font-size: 14px; line-height: 1.45; color: #344054;}
.formula-eq {font-size: 16px; line-height: 1.35; font-weight: 600; color: #101828; letter-spacing: -0.2px; margin: 2px 0 8px;}
.formula-meta {margin-top: auto; padding-top: 8px; border-top: 1px solid #eef2f6; font-size: 13px; line-height: 1.5; color: #667085;}
.callout {margin-top: 12px; background: #fff; border: 1px solid #e5eaf0; border-left: 3px solid #ff8a1f; border-radius: 12px; padding: 12px 14px; color: #344054; font-size: 14px; line-height: 1.5;}
.callout strong {color: #101828; font-weight: 600;}
.table-wrap {width: 100%; overflow: hidden; background: #fff; border: 1px solid #e5eaf0; border-radius: 14px; margin: 0 0 8px;}
table.grid {width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 13px; line-height: 1.4;}
table.grid th, table.grid td {padding: 8px 10px; text-align: left; vertical-align: top; border-bottom: 1px solid #eef2f6; overflow-wrap: break-word; word-wrap: break-word;}
table.grid th {background: #f8fafc; color: #667085; font-size: 12px; font-weight: 600;}
table.grid tr:last-child td {border-bottom: 0;}
table.grid td {color: #172033;}
.insight {background: linear-gradient(160deg, #172033, #24384f); color: #fff; border-radius: 18px; padding: 22px 20px; min-height: 100%; box-shadow: 0 14px 30px rgba(23,32,51,.16);}
.insight .eyebrow {background: rgba(255,255,255,.12); color: #dbeafe;}
.insight h2 {color: white; font-size: 26px; letter-spacing: -0.6px; margin: 6px 0 4px; font-weight: 600; line-height: 1.15;}
.insight .sub {color: #cbd5e1; font-size: 13px; line-height: 1.45; margin: 0 0 12px;}
.prob-row {display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,.1); font-size: 13px; line-height: 1.4;}
.prob-row span {color: #e2e8f0; flex: 0 0 auto;}
.prob-row b {font-weight: 600; text-align: right; max-width: 68%;}
.prob-row:last-child {border-bottom: 0;}
div[data-testid="stSegmentedControl"] {margin: 0 0 8px;}
div[data-testid="stSegmentedControl"] button {font-size: 14px !important;}
[data-testid="stPlotlyChart"] {background: #fff; border: 1px solid #e5eaf0; border-radius: 16px; padding: 4px 4px 0; overflow: hidden;}
.katex-display {overflow-x: auto; overflow-y: hidden; margin: 0.4em 0 !important;}
.table-title {font-size: 14px; font-weight: 600; color: #101828; margin: 12px 0 8px;}
@media (max-width: 1100px) {
  .kpi-row, .formula-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .block-container {padding: 20px 16px 48px;}
  h1 {font-size: 30px;}
  .kpi-value {font-size: 22px;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def household_price_lp(csv_name: str):
    """Shared and per-household programmes. Cached so the view switch does not refit."""
    detail = optimise_prices_detailed(csv_name)
    structural = structural_sensitivity(detail, pct=SHOCK_PCT)
    return detail, structural


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
    labels = [household["label"].replace("Household ", "HH") for household in households]
    fig = go.Figure()
    fig.add_bar(name="Regular", x=labels, y=[household["d_regular"] for household in households], marker_color="#1677ff")
    fig.add_bar(name="Premium", x=labels, y=[household["d_premium"] for household in households], marker_color="#ff8a1f")
    fig.update_layout(barmode="group")
    apply_chart_layout(
        fig,
        height=360,
        yaxis_title="Capsules per buying week",
        left=64,
        right=16,
        top=16,
        bottom=72,
    )
    fig.update_yaxes(gridcolor="#e8edf3", zeroline=False)
    fig.update_xaxes(zeroline=False)
    return fig


def render_table(frame: pd.DataFrame) -> None:
    """Full-width table. Cells wrap, so the page does not grow a horizontal scrollbar."""
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for record in frame.itertuples(index=False):
        cells = "".join(f"<td>{escape('' if value is None else str(value))}</td>" for value in record)
        rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        '<div class="table-wrap"><table class="grid"><thead><tr>'
        + headers
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _pretty_number_text(text: str) -> str:
    rounded = re.sub(r"\d+\.\d+", lambda match: f"{float(match.group()):.2f}", str(text))
    return rounded.replace("<=", "≤").replace(">=", "≥").replace("max  ", "max ")


def _short_menu(text: str) -> str:
    parts = []
    for piece in str(text).split(", "):
        bits = piece.split()
        if len(bits) >= 4 and bits[0] == "Household" and bits[2] == "buys":
            parts.append(f"H{bits[1]} {bits[-1]}")
    return " · ".join(parts) if parts else str(text)


def _fmt2(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if number != number:
        return "—"
    return f"{number:,.2f}"


def section_head(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="section-head"><div class="card-title">{escape(title)}</div>'
        f'<div class="card-sub">{escape(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def _formula_article(kicker: str, blocks: list[tuple[str, str]], meta: list[str]) -> str:
    body = "".join(f'<div class="{kind}">{escape(text)}</div>' for kind, text in blocks)
    meta_html = "<br>".join(escape(line) for line in meta)
    return (
        '<article class="formula-card">'
        f'<div class="formula-kicker">{escape(kicker)}</div>'
        f"{body}"
        f'<div class="formula-meta">{meta_html}</div>'
        "</article>"
    )


def formula_cards(solved: dict, households: list[dict]) -> str:
    coeff = solved["c"]
    parts = []
    if abs(float(coeff[0])) > 1e-12:
        parts.append(f"{float(coeff[0]):.2f}R")
    if abs(float(coeff[1])) > 1e-12:
        parts.append(f"{float(coeff[1]):.2f}P")
    objective = "max " + " + ".join(parts) if parts else "max 0"
    cards = [
        _formula_article(
            "Objective",
            [("formula-eq", objective)],
            ["Weekly revenue from the fixed quantities"],
        )
    ]
    for household in households:
        blocks = []
        for line in household["lines"]:
            if line.get("fixed_threshold") is not None:
                level = float(line["fixed_threshold"])
                blocks.append(("formula-eq", f"P = {level:.2f}"))
                blocks.append(("formula-lead", f"Premium when P ≤ {level:.2f}"))
                blocks.append(("formula-lead", f"Regular when P ≥ {level:.2f}"))
            elif line["kind"] == "switch":
                blocks.append(("formula-lead", "Switches between Regular and Premium"))
                blocks.append(("formula-eq", format_display_equation(line)))
            else:
                blocks.append(("formula-lead", "Stops buying above"))
                blocks.append(("formula-eq", format_display_equation(line)))
        cards.append(
            _formula_article(
                household["label"],
                blocks,
                [
                    f"Regular quantity d = {household['d_regular']:.2f} (n = {household['n_regular']})",
                    f"Premium quantity d = {household['d_premium']:.2f} (n = {household['n_premium']})",
                ],
            )
        )
    return f'<div class="formula-grid">{"".join(cards)}</div>'


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
            f'<div class="prob-row"><span>Binding</span><b>{escape(_pretty_number_text(name.split(" so that ", 1)[-1]))}</b></div>'
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


detail, structural = household_price_lp(DATA_CSV)
shared = detail["shared"]["best"]
labels = [ALL_VIEW] + [item["household"]["label"] for item in detail["per_household"]]

st.markdown(
    '<div class="eyebrow">Whiteboard linear programme</div>'
    "<h1>Revenue-maximising prices</h1>"
    '<p class="lede">Each household buys a fixed quantity of one product, so revenue is linear in the Regular price R and the Premium price P. '
    "A line in that plane decides the product. The shared menu is the feasible assignment with the highest revenue.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="kpi-row">'
    f'<div class="kpi"><div class="kpi-label">Shared Regular price</div><div class="kpi-value">{money(shared["R"])}</div></div>'
    f'<div class="kpi"><div class="kpi-label">Shared Premium price</div><div class="kpi-value">{money(shared["P"])}</div></div>'
    f'<div class="kpi"><div class="kpi-label">Total weekly revenue</div><div class="kpi-value">{money(shared["revenue"])}</div></div>'
    f'<div class="kpi"><div class="kpi-label">Assignment</div><div class="kpi-value compact">{escape(short_assignment(shared["assignment"]))}</div></div>'
    "</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="switch-label">Household</div>', unsafe_allow_html=True)
view = st.segmented_control(
    "Household",
    options=labels,
    default=ALL_VIEW,
    key="lp_view",
    label_visibility="collapsed",
    width="stretch",
    format_func=lambda option: "All households" if option == ALL_VIEW else option,
    wrap=True,
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

st.markdown(
    f'<div class="section-head" style="margin-top:8px"><div class="card-title">{escape(chart_title)}</div>'
    f'<div class="card-sub">{escape(chart_note)}</div></div>',
    unsafe_allow_html=True,
)
left, right = st.columns([1.55, 0.9], gap="medium")
with left:
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False, "responsive": True})
with right:
    st.markdown(result_card(solved, shown, view), unsafe_allow_html=True)
    if solved.get("flat_price"):
        st.caption(solved["flat_price"])

st.markdown(formula_cards(solved, shown), unsafe_allow_html=True)

week6 = detail["frame"].loc[detail["frame"]["T"] == 6]
threshold = next(
    (
        float(line["fixed_threshold"])
        for household in detail["households"]
        for line in household["lines"]
        if line.get("fixed_threshold") is not None
    ),
    None,
)
if not week6.empty and threshold is not None:
    row = week6.iloc[0]
    st.markdown(
        '<div class="callout"><strong>Household 2.</strong> '
        f"The line is the separating threshold P = {threshold:.2f}, because OLS misclassified week 6 "
        f"(R = {row['P_Regular']:.0f}, P = {row['P_Premium']:.0f}).</div>",
        unsafe_allow_html=True,
    )

section_head(
    "Constraints",
    "Every side of each switching line, the price box, and the objective of the programme in view. "
    "Binding is evaluated at that programme's optimum. A side marked “not in this programme” is the other side of the line.",
)
constraints = constraint_table(solved, detail["households"] if view == ALL_VIEW else shown).copy()
constraints["Constraint"] = constraints["Constraint"].map(_pretty_number_text)
constraints = constraints.rename(columns={"In-sample accuracy": "Accuracy"})
render_table(constraints)

section_head(
    "Average weekly demand",
    "d is the average number of capsules on weeks when that household bought the product. "
    "Weeks with a zero stay in the sample and are not part of this average.",
)
demand_left, demand_right = st.columns([1, 1.15], gap="medium")
with demand_left:
    demand = demand_table(detail["households"])
    styled = demand.copy()
    styled["d Regular"] = styled["d Regular"].map(lambda value: f"{value:.2f}")
    styled["d Premium"] = styled["d Premium"].map(lambda value: f"{value:.2f}")
    render_table(styled)
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

section_head(
    "Sensitivity, ±10%",
    f"{shock_note} The next chart moves the household's own optimal price by {shock_label} and recomputes revenue from the same lines.",
)
if not line_shocks.empty:
    st.plotly_chart(structural_sensitivity_figure(line_shocks), width="stretch", config={"displayModeBar": False, "responsive": True})
    show = line_shocks[["Household", "parameter", "shock", "R_opt", "P_opt", "max_revenue", "delta_revenue", "assignment"]].copy()
    show["parameter"] = show["parameter"].map(lambda text: _pretty_number_text(str(text).replace("threshold P = ", "threshold ")))
    show["assignment"] = show["assignment"].map(_short_menu)
    for column in ("R_opt", "P_opt", "max_revenue", "delta_revenue"):
        show[column] = show[column].map(_fmt2)
    show.columns = ["Household", "Term", "Shock", "R", "P", "Revenue", "Δ revenue", "Assignment"]
    render_table(show)
if not price_moves.empty:
    st.plotly_chart(price_move_figure(price_moves), width="stretch", config={"displayModeBar": False, "responsive": True})

section_head(
    "Scenario comparison and regret",
    "The scenario is a Regular / Premium price pair. Regret is a household's own LP revenue minus revenue at that menu. "
    "The cell on a household's own optimum is zero.",
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
_MENU = {
    "Shared LP optimum": "Shared LP",
    "Household 1 LP optimum": "HH1 LP",
    "Household 2 LP optimum": "HH2 LP",
    "Household 3 LP optimum": "HH3 LP",
    "Scenario prices": "Scenario",
}
rev_pivot = regret_view.pivot(index="Household", columns="Menu", values="revenue").rename(columns=_MENU)
reg_pivot = regret_view.pivot(index="Household", columns="Menu", values="regret").rename(columns=_MENU)
st.markdown('<div class="table-title">Revenue by scenario (€)</div>', unsafe_allow_html=True)
render_table(rev_pivot.reset_index().map(_fmt2))
st.markdown('<div class="table-title">Regret versus that household\'s LP optimum (€)</div>', unsafe_allow_html=True)
render_table(reg_pivot.reset_index().map(_fmt2))


def _latex_line(slope: float, icept: float, digits: int = 4) -> str:
    if abs(slope) < 1e-8:
        return rf"P = {icept:.{digits}f}"
    sign = "+" if icept >= 0 else "-"
    return rf"P = {slope:.{digits}f}\, R {sign} {abs(icept):.{digits}f}"


def _latex_matrix(rows, digits: int | None = None) -> str:
    body = []
    for row in rows:
        cells = []
        for value in row:
            number = float(value)
            if digits is None and abs(number - round(number)) < 1e-8:
                cells.append(str(int(round(number))))
            else:
                cells.append(f"{number:.{4 if digits is None else digits}f}")
        body.append(" & ".join(cells))
    return r"\begin{bmatrix}" + r" \\ ".join(body) + r"\end{bmatrix}"


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


def render_methodology(detail: dict) -> None:
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
        '<div class="section-head"><div class="card-title">Model & methodology</div>'
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
    st.markdown("The next section does this arithmetic for Household 1, then for all three households.")
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
    st.markdown(
        "- **Scenario.** A Regular / Premium pair — by default the average prices in the table — is scored with the same lines. "
        "Menus in the comparison are the shared optimum, each household's own optimum, and that scenario."
    )
    st.latex(r"\mathrm{regret}_i(\mathrm{menu}) = \mathrm{revenue}_i(\mathrm{own\ optimum}) - \mathrm{revenue}_i(\mathrm{menu})")
    st.markdown("Regret is zero on a household's own optimum, and positive when another menu earns them less.")

    st.markdown("#### Limitations")
    cap = float(bounds["R_upper"])
    cap_label = f"€{cap:.0f}" if abs(cap - round(cap)) < 1e-6 else money(cap)
    if abs(shared["R"] - cap) <= 1e-6:
        st.markdown(
            f"- R sits at the {cap_label} cap because that's the highest observed Regular price, and the optimum would follow a higher cap."
        )
    else:
        st.markdown(
            f"- The optimum Regular price is {money(shared['R'])}, inside a box whose top is the highest observed Regular price, {money(cap)}."
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


def _coef(value: float) -> str:
    return f"{value:.4f}"


def render_line_origins(detail: dict) -> None:
    """Where each switching line comes from. Every figure is computed from `detail`."""
    frame = detail["frame"]
    households = detail["households"]
    fits = [ols_regression(frame, household) for household in households]
    by_label = {household["label"]: fit for household, fit in zip(households, fits)}
    focus_label = "Household 1" if "Household 1" in by_label else households[0]["label"]
    focus = by_label[focus_label]
    focus_household = next(item for item in households if item["label"] == focus_label)

    section_head(
        "Where the line equations come from",
        "How a week becomes a 0 or a 1, how least squares turns that into a line, and why another line through the same gap is also allowed.",
    )

    st.markdown("#### How ordinary least squares draws the line")
    st.markdown(
        "For each household, label every week with a number y. "
        "**Premium is 1 and Regular is 0.** "
        "Household 3 never buys Regular, so there y is 1 on a week they bought something and 0 on a week they bought nothing."
    )
    st.markdown(
        "Least squares then finds the intercept a and the slopes b and c that make"
    )
    st.latex(r"y = a + b R + c P")
    st.markdown(
        "as close as it can to those labels. Read the fitted value as a rough probability of Premium "
        "(or of buying, for Household 3). The line on the chart is where that fitted value is one half:"
    )
    st.latex(r"a + b R + c P = 0.5")
    st.markdown("Solve that for P:")
    st.latex(r"P = \frac{0.5 - a - b R}{c} = m R + k")
    st.latex(r"m = -\frac{b}{c}, \qquad k = \frac{0.5 - a}{c}")

    st.markdown(f"#### {focus_label}, one week at a time")
    st.markdown(
        f"These are the {len(focus['rows'])} weeks in {focus_label}'s regression. "
        f"y = 1 means {focus['positive_label']}."
    )
    week_table = pd.DataFrame([
        {
            "Week": row["week"],
            "R": f"{row['R']:.0f}",
            "P": f"{row['P']:.0f}",
            "Choice": "nothing" if row["observed"] == "None" else row["observed"],
            "y": row["y"],
        }
        for row in focus["rows"]
    ])
    render_table(week_table)
    st.markdown("Least squares on that table gives")
    st.latex(rf"a = {_coef(focus['a'])}, \quad b = {_coef(focus['b'])}, \quad c = {_coef(focus['c'])}")
    st.markdown("Turn the coefficients into the slope and intercept of the 0.5 contour:")
    st.latex(
        rf"m = -\frac{{b}}{{c}} = {_coef(focus['m'])}, \qquad "
        rf"k = \frac{{0.5 - a}}{{c}} = {_coef(focus['k'])}"
    )
    st.latex(_latex_line(focus["m"], focus["k"], digits=4))
    st.markdown("The chart rounds that to two decimals:")
    st.latex(_latex_line(focus["m"], focus["k"], digits=2))

    example = next((row for row in focus["rows"] if row["week"] == EXAMPLE_WEEK), focus["rows"][0])
    called = predict_product(
        {"lines": [focus["line"]], "only_product": focus_household.get("only_product")},
        example["R"],
        example["P"],
    )
    side = "above" if example["fitted"] >= 0.5 else "below"
    observed = "nothing" if example["observed"] == "None" else example["observed"]
    called_text = "nothing" if called == "None" else called
    match = "which matches" if called == example["observed"] else "which does not match"
    st.markdown(
        f"Week {example['week']} has R = {example['R']:.0f} and P = {example['P']:.0f}. Plug those prices in:"
    )
    st.latex(
        rf"\hat{{y}} = a + b \cdot {example['R']:.0f} + c \cdot {example['P']:.0f} = {example['fitted']:.4f}"
    )
    st.markdown(
        f"{example['fitted']:.4f} is {side} one half, so the line says **{called_text}**. "
        f"{focus_label} bought {observed} that week, {match}."
    )

    with st.expander("The same fit in matrix form"):
        st.markdown(
            "Stack a column of ones, the Regular prices, and the Premium prices into X, "
            "and the labels into y. The least-squares coefficients are"
        )
        st.latex(r"\hat\beta = (X^\top X)^{-1} X^\top y")
        st.latex(rf"X = {_latex_matrix(focus['X'])}")
        st.latex(rf"y = {_latex_matrix([[value] for value in focus['y']])}")
        st.latex(rf"X^\top X = {_latex_matrix(focus['XtX'])}, \qquad X^\top y = {_latex_matrix([[value] for value in focus['Xty']])}")
        st.latex(
            rf"\hat\beta = {_latex_matrix([[value] for value in focus['beta']], digits=4)} "
            rf"= \begin{{bmatrix}} a \\ b \\ c \end{{bmatrix}}"
        )

    st.markdown("#### All three households")
    st.markdown(
        "The same regression for each household. Accuracy is how many of the "
        f"{fits[0]['n']} weeks the fitted line classifies correctly. "
        "Household 2's row is the OLS line, not the threshold the programme uses."
    )
    coef_rows = []
    for household, fit in zip(households, fits):
        coef_rows.append({
            "Household": household["label"],
            "y = 1": fit["positive_label"],
            "a": _coef(fit["a"]),
            "b": _coef(fit["b"]),
            "c": _coef(fit["c"]),
            "m": _coef(fit["m"]),
            "k": _coef(fit["k"]),
            "Accuracy": f"{fit['correct']}/{fit['n']}",
        })
    render_table(pd.DataFrame(coef_rows))

    threshold_household = next(
        (household for household in households if any(line.get("fixed_threshold") is not None for line in household["lines"])),
        None,
    )
    if threshold_household is not None:
        threshold_fit = by_label[threshold_household["label"]]
        level = next(
            float(line["fixed_threshold"])
            for line in threshold_household["lines"]
            if line.get("fixed_threshold") is not None
        )
        misses = _ols_misses(frame, threshold_household)
        if misses:
            described = "; ".join(
                f"week {miss['week']} (R = {miss['R']:.0f}, P = {miss['P']:.0f}) was {miss['observed']}, "
                f"and the OLS line called it {miss['predicted']}"
                for miss in misses
            )
            miss_fit = next(
                (row["fitted"] for row in threshold_fit["rows"] if row["week"] == misses[0]["week"]),
                None,
            )
            fitted_bit = f" The fitted value that week is {miss_fit:.4f}." if miss_fit is not None else ""
            st.markdown(
                f'<div class="callout"><strong>{escape(threshold_household["label"])}.</strong> '
                f"OLS gets {threshold_fit['correct']}/{threshold_fit['n']}. It misclassifies {escape(described)}."
                f"{escape(fitted_bit)} "
                f"The programme uses the separating threshold P = {level:.2f} instead, "
                f"which scores {_accuracy_label_local(threshold_household)}.</div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Why other lines are also valid")
    st.markdown(
        "The weeks fall into two groups with a gap between them, so many lines split the sample cleanly. "
        "A line drawn through that gap — by hand, or by a max-margin rule — can score as well as OLS. "
        "OLS is pulled toward every week, which is why Household 1's fitted line is steeper than a line that only has to sit in the gap."
    )
    st.markdown("These three lines are of that kind:")
    for household, fit in zip(households, fits):
        slope, icept = ALTERNATIVE_CONTOURS[household["label"]]
        st.latex(rf"\text{{{household['label']}}} \quad {_contour_latex(slope, icept)}")

    alt_lines = {
        household["label"]: contour_line(*ALTERNATIVE_CONTOURS[household["label"]], fit["kind"])
        for household, fit in zip(households, fits)
    }
    alt_rows = []
    for household, fit in zip(households, fits):
        correct, n = classify_accuracy(frame, household, alt_lines[household["label"]])
        slope, icept = ALTERNATIVE_CONTOURS[household["label"]]
        sign = "−" if icept < 0 else "+"
        alt_rows.append({
            "Household": household["label"],
            "Line": f"P = {slope:.4f}R {sign} {abs(icept):.2f}",
            "Accuracy": f"{correct}/{n}",
        })
    render_table(pd.DataFrame(alt_rows))

    clones = [{**household, "lines": [alt_lines[household["label"]]]} for household in households]
    alternative = best_programme(clones, detail["bounds"])["best"]
    shared = detail["shared"]["best"]
    if alternative is None:
        st.markdown("With these lines the shared programme has no feasible price pair inside the box.")
    else:
        st.markdown(
            f"Put into the same linear programme — same quantities, same price box — they give "
            f"**{alternative['assignment_label']}**, at R = {money(alternative['R'])}, "
            f"P = {money(alternative['P'])}, weekly revenue {money(alternative['revenue'])}. "
            f"The OLS menu, with Household 2 on its threshold, is {money(shared['revenue'])} "
            f"at P = {money(shared['P'])}."
        )
    focus_alt_m = ALTERNATIVE_CONTOURS[focus_label][0]
    st.markdown(
        f"{focus_label}'s OLS slope is {focus['m']:.2f}. The other line's slope is {focus_alt_m:.2f}. "
        "Every week tugs the OLS line, including weeks far from the boundary. "
        "A line placed in the gap only has to keep the two groups apart, so it can be much flatter."
    )
    st.markdown(
        "OLS is a defensible choice because it is reproducible: the same table always returns the same coefficients. "
        "A hand-placed line depends on which gap you decide to sit in."
    )


def _contour_latex(slope: float, icept: float) -> str:
    """Classmate lines keep four decimals on the slope and two on the intercept."""
    sign = "+" if icept >= 0 else "-"
    return rf"P = {slope:.4f}\, R {sign} {abs(icept):.2f}"


render_methodology(detail)
render_line_origins(detail)

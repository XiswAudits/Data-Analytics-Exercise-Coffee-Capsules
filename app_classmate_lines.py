"""Streamlit page for the whiteboard price linear programme on hand-placed lines.

This is a second entry point. The root `app.py` still uses the OLS lines
(Household 2 on P = 77.5). Here the constraints are the given separating
lines. Quantities, bounds, and the solver still come from `price_optimisation.py`.
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
    classify_accuracy,
    constraint_table,
    format_display_equation,
    household_figure,
    ols_regression,
    ols_switch_line,
    optimise_prices_detailed,
    predict_product,
    price_move_figure,
    price_move_overview,
    regret_figure,
    regret_table,
    shared_figure,
    structural_sensitivity,
    structural_sensitivity_figure,
)

SHOCK_PCT = 0.10
EXAMPLE_WEEK = 4

DATA_CSV = str(Path(__file__).resolve().parent / "coffee_capsules_data.csv")
ALL_VIEW = "All households (shared prices)"

st.set_page_config(page_title="Revenue-maximising prices — given lines", page_icon="☕", layout="wide", initial_sidebar_state="collapsed")

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
.table-title {display: inline-block; font-size: 14px; font-weight: 600; color: #101828; margin: 12px 0 8px;}
.overview-card {margin-top: 12px;}
.overview-house {margin-top: 14px; padding-top: 12px; border-top: 1px solid #eef2f6;}
.overview-house:first-of-type {margin-top: 12px;}
.overview-kicker {font-size: 12px; font-weight: 600; color: #4b6380; letter-spacing: 0.01em; margin-bottom: 4px;}
.overview-best {font-size: 14px; line-height: 1.5; color: #101828; margin: 0 0 8px;}
.overview-moves {margin: 0; padding-left: 18px;}
.overview-moves li {font-size: 13px; line-height: 1.5; color: #344054; margin: 0 0 6px;}
.overview-takeaway {margin: 14px 0 0; padding-top: 10px; border-top: 1px solid #eef2f6; font-size: 14px; line-height: 1.5; color: #101828;}
[data-testid="stTooltipIcon"] svg {width: 14px; height: 14px; stroke: #98a2b3;}
.table-caption {font-size: 13px; line-height: 1.45; color: #667085; margin: 2px 0 8px;}
.compare-grid {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 12px;}
.compare-k {font-size: 12px; line-height: 1.35; color: #667085; font-weight: 500;}
.compare-v {margin-top: 4px; font-size: 16px; line-height: 1.3; letter-spacing: -0.2px; font-weight: 600; color: #101828;}
@media (max-width: 1100px) {
  .kpi-row, .formula-grid, .compare-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .block-container {padding: 20px 16px 48px;}
  h1 {font-size: 30px;}
  .kpi-value {font-size: 22px;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def household_price_lp(csv_name: str):
    """Shared and per-household programmes on the given lines, plus the OLS menu for comparison."""
    detail = optimise_prices_detailed(csv_name, line_source="classmate")
    ols = optimise_prices_detailed(csv_name, line_source="ols")
    structural = structural_sensitivity(detail, pct=SHOCK_PCT)
    return detail, structural, ols


def money(value: float) -> str:
    return f"€{value:,.2f}"


def revenue_versus(gap: float, other: str) -> str:
    """Say how this page's revenue sits relative to another solved menu."""
    if abs(gap) < 0.005:
        return f"This page's weekly revenue matches the {other}."
    direction = "below" if gap < 0 else "above"
    return f"This page's weekly revenue is {money(abs(gap))} {direction} the {other}."


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


def render_table(frame: pd.DataFrame, widths: list[str] | None = None) -> None:
    """Full-width table. Cells wrap, so the page does not grow a horizontal scrollbar."""
    columns = "".join(f'<col style="width:{escape(width)}">' for width in widths) if widths else ""
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for record in frame.itertuples(index=False):
        cells = "".join(f"<td>{escape('' if value is None else str(value))}</td>" for value in record)
        rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        '<div class="table-wrap"><table class="grid">'
        + columns
        + "<thead><tr>"
        + headers
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _pretty_number_text(text: str) -> str:
    rounded = re.sub(r"\d+\.\d+", lambda match: f"{float(match.group()):.2f}", str(text))
    return rounded.replace("<=", "≤").replace(">=", "≥").replace("max  ", "max ")


def _pretty_constraint(text: str) -> str:
    """Keep four decimals on a hand-placed slope such as 0.6667 or -0.0502."""
    return str(text).replace("<=", "≤").replace(">=", "≥").replace("max  ", "max ")


def _hh_tag(label: str) -> str:
    return "HH" + str(label).split()[-1]


def _assignment_map(text: str) -> dict[str, str]:
    found = {}
    for piece in str(text).split(", "):
        bits = piece.split()
        if len(bits) >= 4 and bits[0] == "Household" and bits[2] == "buys":
            found[f"Household {bits[1]}"] = bits[-1]
    return found


def _shock_factor(shock: str) -> float:
    raw = str(shock).replace("−", "-").replace("%", "").strip()
    return 1.0 + float(raw) / 100.0


def _switch_phrase(changes: list[tuple[str, str, str]]) -> str:
    bits = []
    for label, old, new in changes:
        who = _hh_tag(label)
        if new == "None":
            bits.append(f"{who} drops out")
        elif old == "None":
            bits.append(f"{who} buys {new}")
        else:
            bits.append(f"{who} switches to {new}")
    return ", ".join(bits)


def _what_happens(row, detail: dict) -> str:
    """One line on how this shocked line moved the shared menu.

    Built from the base programme and the re-solved row. An assignment is
    named only when it is not the base assignment.
    """
    base = detail["shared"]["best"]
    label = str(row.Household)
    tag = _hh_tag(label)
    household = next(item for item in detail["households"] if item["label"] == label)
    line = next((item for item in household["lines"] if item.get("given_m") is not None), household["lines"][0])
    slope = float(line.get("given_m", 0.0))
    icept = float(line.get("given_k", 0.0))
    factor = _shock_factor(str(row.shock))
    new_slope, new_icept = slope, icept
    if "slope" in str(row.parameter):
        new_slope *= factor
    else:
        new_icept *= factor
    base_r = float(base["R"])
    base_p = float(base["P"])
    new_r = float(row.R_opt)
    new_p = float(row.P_opt)
    if new_r != new_r or new_p != new_p:
        return f"{tag} line makes the programme infeasible"
    base_level = slope * base_r + icept
    new_level = new_slope * base_r + new_icept
    rose = new_level > base_level + 1e-6
    fell = new_level < base_level - 1e-6
    base_assign = dict(base["assignment"])
    new_assign = _assignment_map(str(row.assignment))
    if not new_assign:
        return f"{tag} line makes the programme infeasible"
    product = base_assign[label]
    new_product = new_assign.get(label, product)
    is_cap = product != "Regular"
    role = "cap" if is_cap else "floor"
    prices_same = abs(new_r - base_r) < 0.005 and abs(new_p - base_p) < 0.005
    changed = [
        (name, base_assign[name], new_assign[name])
        for name in base_assign
        if new_assign.get(name) != base_assign[name]
    ]
    binding = any(label in name and "so that" in name for name in (base.get("tight") or []))
    if prices_same and not changed:
        if not binding:
            return f"{tag} line not binding, no change"
        return f"{tag} line moves, optimum unchanged"

    if role == "floor" and rose and new_level > base_p + 0.02 and product == "Regular" and new_product == "Premium":
        sentence = f"{tag} floor rises above the cap, so {tag} switches to Premium"
        others = [item for item in changed if item[0] != label]
        if others:
            sentence += "; " + _switch_phrase(others)
        if not prices_same:
            sentence += f", P to {new_p:.2f}"
        return sentence

    if role == "cap" and not changed:
        if rose and new_p > base_p + 0.02:
            return f"{tag} Premium cap rises, P can go up to {new_p:.2f}"
        if fell and new_p < base_p - 0.02:
            if abs(new_r - base_r) > 0.02:
                return f"{tag} Premium cap falls, R to {new_r:.2f}, P cut to {new_p:.2f}"
            return f"{tag} Premium cap falls, P cut to {new_p:.2f}"

    if product != "None" and new_product == "None" and abs(new_p - base_p) <= 0.05:
        others = [item for item in changed if item[0] != label]
        tail = "; " + _switch_phrase(others) if others else ""
        return f"{tag} drops out, cheaper to lose it than cut P{tail}"

    if role == "cap" and fell and new_product == product:
        if abs(new_r - base_r) > 0.02 and abs(new_p - base_p) > 0.02:
            move = "R and P cut"
        elif abs(new_p - base_p) > 0.02:
            move = f"P cut to {new_p:.2f}"
        elif abs(new_r - base_r) > 0.02:
            move = f"R cut to {new_r:.2f}"
        else:
            move = "prices barely move"
        sentence = f"{tag} cap falls, {move}"
        if changed:
            sentence += "; " + _switch_phrase(changed)
        return sentence

    direction = "rises" if rose else "falls" if fell else "shifts"
    sentence = f"{tag} {role} {direction}"
    if not prices_same:
        sentence += f", P to {new_p:.2f}"
    if changed:
        sentence += "; " + _switch_phrase(changed)
    return sentence


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


PRICE_MOVE_TIP = (
    "Each bar starts from that household's best prices, moves one price by 10%, and shows the change in revenue you (the seller) earn from it. "
    "Cutting a price earns less from buyers of that product. Raising it too far makes the household switch or stop buying, which costs the most. "
    "No bar = that price doesn't affect it."
)
REVENUE_SCENARIO_TIP = (
    "Each column is one set of prices: the best prices for one household alone (HH LP), the benchmark Scenario prices, or the one Shared LP price pair for everyone. "
    "Each cell is the revenue that household brings in at those prices. Computed from the data; only the Scenario column changes if you change the scenario prices."
)
REGRET_TIP = (
    "Regret = how much you miss compared with that household's best case: its best revenue minus the revenue at these prices. "
    "0 means these prices are already best for it; bigger = worse. Shared LP has the smallest regret overall."
)


def info_title(title: str, tip: str) -> None:
    """Title row with Streamlit's small hover tooltip, in the same type as the tables."""
    st.markdown(
        f'<div class="table-title">{escape(title)}</div>',
        unsafe_allow_html=True,
        help=tip,
        width="content",
    )


def chart_with_tip(fig: go.Figure, title: str, tip: str) -> None:
    """Show a chart whose title sits beside the tooltip, not inside the plot."""
    fig.update_layout(title=None, margin_t=16)
    info_title(title, tip)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})


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
            f'<div class="prob-row"><span>Binding</span><b>{escape(_pretty_constraint(name.split(" so that ", 1)[-1]))}</b></div>'
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


detail, structural, ols_detail = household_price_lp(DATA_CSV)
shared = detail["shared"]["best"]
ols_shared = ols_detail["shared"]["best"]
labels = [ALL_VIEW] + [item["household"]["label"] for item in detail["per_household"]]

st.markdown(
    '<div class="eyebrow">Hand-placed separating lines</div>'
    "<h1>Revenue-maximising prices</h1>"
    '<p class="lede">Each household buys a fixed quantity of one product, so revenue is linear in the Regular price R and the Premium price P. '
    "The lines that decide the product are given: they were placed through the gap between weeks, not fitted by least squares. "
    "The shared menu is the feasible assignment with the highest revenue.</p>",
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

revenue_gap = shared["revenue"] - ols_shared["revenue"]
st.markdown(
    '<div class="card" style="margin:12px 0 4px">'
    '<div class="card-title">Compared with the OLS lines</div>'
    '<div class="card-sub">Same quantities and the same price box. Both programmes are solved from the table; neither optimum is typed in.</div>'
    '<div class="compare-grid">'
    f'<div><div class="compare-k">Hand-placed R / P</div><div class="compare-v">{money(shared["R"])} / {money(shared["P"])}</div></div>'
    f'<div><div class="compare-k">Hand-placed revenue</div><div class="compare-v">{money(shared["revenue"])}</div></div>'
    f'<div><div class="compare-k">OLS R / P</div><div class="compare-v">{money(ols_shared["R"])} / {money(ols_shared["P"])}</div></div>'
    f'<div><div class="compare-k">OLS revenue</div><div class="compare-v">{money(ols_shared["revenue"])}</div></div>'
    "</div>"
    f'<div class="card-sub" style="margin-top:10px">Assignment here: {escape(short_assignment(shared["assignment"]))}. '
    f'Assignment on the OLS lines: {escape(short_assignment(ols_shared["assignment"]))}. '
    f'{escape(revenue_versus(revenue_gap, "OLS revenue"))}.</div>'
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
constraints["Constraint"] = constraints["Constraint"].map(_pretty_constraint)
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
    shock_note = f"Each line's slope m and intercept k are moved by {shock_label} and the shared programme is solved again."
else:
    shock_note = f"{view}'s slope m and intercept k are moved by {shock_label} and the shared programme is solved again. Only that household is shown."

section_head(
    "Sensitivity, ±10%",
    f"{shock_note} The next chart moves the household's own optimal price by {shock_label} and recomputes revenue from the same lines.",
)
if not line_shocks.empty:
    st.plotly_chart(structural_sensitivity_figure(line_shocks), width="stretch", config={"displayModeBar": False, "responsive": True})
    st.markdown(
        f'<p class="table-caption">Each line\'s slope or intercept moved by ±{SHOCK_PCT:.0%}. '
        "The LP is re-solved; the revenue change is versus the base.</p>",
        unsafe_allow_html=True,
    )
    show = line_shocks[["Household", "parameter", "shock", "R_opt", "P_opt", "max_revenue", "delta_revenue", "assignment"]].copy()
    show["What happens"] = [_what_happens(row, detail) for row in show.itertuples(index=False)]
    show["Household"] = show["Household"].map(_hh_tag)
    show["parameter"] = show["parameter"].map(lambda text: _pretty_number_text(str(text).replace("threshold P = ", "threshold ")))
    show = show.drop(columns=["assignment"])
    for column in ("R_opt", "P_opt", "max_revenue", "delta_revenue"):
        show[column] = show[column].map(_fmt2)
    show.columns = ["Household", "Term", "Shock", "R", "P", "Revenue", "Δ revenue", "What happens"]
    render_table(show, widths=["8%", "12%", "8%", "8%", "8%", "10%", "11%", "35%"])
if not price_moves.empty:
    chart_with_tip(
        price_move_figure(price_moves),
        "Revenue when that household's price moves ±10%",
        PRICE_MOVE_TIP,
    )
    st.markdown(price_move_overview(detail, price_moves), unsafe_allow_html=True)

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
chart_with_tip(
    regret_figure(regret_view),
    "Regret versus each household's own optimum",
    REGRET_TIP,
)
_MENU = {
    "Shared LP optimum": "Shared LP",
    "Household 1 LP optimum": "HH1 LP",
    "Household 2 LP optimum": "HH2 LP",
    "Household 3 LP optimum": "HH3 LP",
    "Scenario prices": "Scenario",
}
rev_pivot = regret_view.pivot(index="Household", columns="Menu", values="revenue").rename(columns=_MENU)
reg_pivot = regret_view.pivot(index="Household", columns="Menu", values="regret").rename(columns=_MENU)
info_title("Revenue by scenario (€)", REVENUE_SCENARIO_TIP)
render_table(rev_pivot.reset_index().map(_fmt2))
info_title("Regret versus that household's LP optimum (€)", REGRET_TIP)
render_table(reg_pivot.reset_index().map(_fmt2))


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


def render_methodology(detail: dict, ols_detail: dict) -> None:
    """End-to-end account of this programme. Every figure is computed."""
    frame = detail["frame"]
    households = detail["households"]
    bounds = detail["bounds"]
    shared = detail["shared"]["best"]
    ols_shared = ols_detail["shared"]["best"]
    cases = detail["shared"]["cases"]
    n_weeks = int(frame["T"].nunique())
    n_households = len(households)
    n_feasible = sum(1 for case in cases if case["feasible"])

    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-head"><div class="card-title">Model & methodology</div>'
        '<div class="card-sub">How the prices on this page were obtained. The lines are given; the optimum is solved.</div></div>',
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

    st.markdown("#### The separating lines")
    st.markdown(
        "Which product they buy is a straight line in the R–P plane, written P = m R + k. "
        "These three lines are **given**. They were placed by hand through the gap between the two groups of weeks. "
        "They are not an ordinary-least-squares fit, and the page does not estimate m or k."
    )
    st.latex(r"P = m R + k")
    st.markdown(
        "Premium, for a household that buys both products, is the side **on or below** the line. Regular is the side above it. "
        "Household 3 never buys Regular, so the same rule means they keep buying Premium on or below the line and stop above it. "
        "The next section checks that reading against every week."
    )
    for household in households:
        line = household["lines"][0]
        slope, icept = float(line["given_m"]), float(line["given_k"])
        accuracy = _accuracy_label_local(household)
        st.markdown(f"**{household['label']}** — in-sample accuracy {accuracy}.")
        st.latex(_contour_latex(slope, icept))
        if line["kind"] == "stop":
            st.markdown("Buys Premium on or below this line, and buys nothing above it.")
        else:
            st.markdown("Premium on or below this line, Regular above it.")

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
    st.markdown(
        f"The same programme on the OLS lines (Household 2 on its threshold) is "
        f"**{ols_shared['assignment_label']}**, at R = {money(ols_shared['R'])}, "
        f"P = {money(ols_shared['P'])}, weekly revenue {money(ols_shared['revenue'])}. "
        f"{revenue_versus(shared['revenue'] - ols_shared['revenue'], 'OLS revenue')}."
    )

    st.markdown("#### Sensitivity, scenarios, and regret")
    st.markdown(
        f"- **Line shocks.** Each given slope m and each given intercept k is moved by {SHOCK_PCT:.0%} of itself, "
        "and the shared programme is solved again. A negative slope is scaled the same way. "
        "That shows whether the menu and the revenue depend on a fragile coefficient."
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
        f"- **The sample is small:** {n_weeks} weeks and {n_households} households. "
        "The optimum is a reading of this table, not a forecast."
    )
    st.markdown(
        "- **The lines are a judgement.** Many lines sit in the gap and separate the weeks. "
        "Ordinary least squares would return one of them, and it would be reproducible. "
        "A hand-placed line depends on which gap you decide to sit in. This page uses one such reading."
    )
    st.markdown(
        "- **Quantities are fixed.** Inside a region, selling at a higher price does not reduce d. That is why revenue is linear, and why a price on the edge of a region looks attractive. It is not a demand curve."
    )


def _accuracy_label_local(household: dict) -> str:
    n = household["n_regular"] + household["n_premium"] + household["n_none"]
    correct = int(round(household["accuracy"] * n))
    return f"{correct}/{n}"


def _observed_product(row: pd.Series, household: dict) -> str:
    prefix = household["prefix"]
    if float(row[f"{prefix}_Regular"]) > 0:
        return "Regular"
    if float(row[f"{prefix}_Premium"]) > 0:
        return "Premium"
    return "None"


def _side_phrase(line: dict, regular: float, premium: float) -> str:
    boundary = float(line["given_m"]) * regular + float(line["given_k"])
    return "on or below" if premium <= boundary + 1e-9 else "above"


def render_given_lines(detail: dict, ols_detail: dict) -> None:
    """Where the constraints come from. Every figure is computed from the table."""
    frame = detail["frame"]
    households = detail["households"]
    fits = {household["label"]: ols_regression(frame, household) for household in households}
    focus_label = "Household 1" if any(item["label"] == "Household 1" for item in households) else households[0]["label"]
    focus = next(item for item in households if item["label"] == focus_label)
    focus_line = focus["lines"][0]

    section_head(
        "Where these line equations come from",
        "The lines are written down in advance and then checked against the weeks. They are not a least-squares fit.",
    )

    st.markdown("#### Given, not fitted")
    st.markdown(
        "Each constraint is a line someone placed through the gap between two groups of weeks:"
    )
    st.latex(r"P = m R + k")
    st.markdown(
        "m and k are inputs. The table is used only to check which side each week falls on, "
        "and to read the fixed quantities and the price box. "
        "Ordinary least squares is a different reading of the same weeks: every week pulls the line, "
        "so its slope need not match a line that only has to keep the groups apart."
    )

    st.markdown("#### Which side is which product")
    st.markdown(
        "A price is on or below the line when P is at most m R + k. "
        "That side is Premium for Household 1 and Household 2, and it is still buying Premium for Household 3. "
        "The other side is Regular, or no purchase for Household 3."
    )
    st.latex(r"P \le m R + k")
    st.markdown("is the same statement as a score of at least one half:")
    st.latex(r"\mathrm{score} = (0.5 + k) + m R - P")
    st.latex(r"\mathrm{score} \ge 0.5 \iff P \le m R + k")
    st.markdown("The linear programme uses that half-space. The chart draws the line itself.")

    st.markdown("#### Checked on every week")
    st.markdown(
        "For each household the line is scored against all "
        f"{int(frame['T'].nunique())} weeks. Accuracy is how many of those weeks land on the product the line implies."
    )
    summary = []
    for household in households:
        line = household["lines"][0]
        correct, n = classify_accuracy(frame, household, line)
        slope, icept = float(line["given_m"]), float(line["given_k"])
        sign = "−" if icept < 0 else "+"
        summary.append({
            "Household": household["label"],
            "Line": f"P = {slope:.4f}R {sign} {abs(icept):.2f}",
            "Rule": "Premium on or below" if line["kind"] == "switch" else "buys on or below, stops above",
            "Accuracy": f"{correct}/{n}",
        })
    render_table(pd.DataFrame(summary))

    st.markdown(f"#### {focus_label}, one week at a time")
    week_rows = []
    for _, row in frame.iterrows():
        regular = float(row["P_Regular"])
        premium = float(row["P_Premium"])
        observed = _observed_product(row, focus)
        called = predict_product(focus, regular, premium)
        week_rows.append({
            "Week": int(row["T"]),
            "R": f"{regular:.0f}",
            "P": f"{premium:.0f}",
            "Observed": "nothing" if observed == "None" else observed,
            "Side": _side_phrase(focus_line, regular, premium),
            "Line says": "nothing" if called == "None" else called,
        })
    render_table(pd.DataFrame(week_rows))
    correct, n = classify_accuracy(frame, focus, focus_line)
    st.markdown(
        f"The line and the observed product agree on **{correct}** of **{n}** weeks."
    )

    example = frame.loc[frame["T"] == EXAMPLE_WEEK]
    example_row = example.iloc[0] if not example.empty else frame.iloc[0]
    regular = float(example_row["P_Regular"])
    premium = float(example_row["P_Premium"])
    slope = float(focus_line["given_m"])
    icept = float(focus_line["given_k"])
    boundary = slope * regular + icept
    observed = _observed_product(example_row, focus)
    called = predict_product(focus, regular, premium)
    observed_text = "nothing" if observed == "None" else observed
    called_text = "nothing" if called == "None" else called
    relation = "on or below" if premium <= boundary + 1e-9 else "above"
    match = "which matches" if called == observed else "which does not match"
    st.markdown(
        f"Week {int(example_row['T'])} has R = {regular:.0f} and P = {premium:.0f}. "
        f"The {focus_label} line at that Regular price is"
    )
    st.latex(rf"m \cdot {regular:.0f} + k = {boundary:.4f}")
    st.markdown(
        f"P = {premium:.0f} is {relation} {boundary:.4f}, so the line says **{called_text}**. "
        f"{focus_label} bought {observed_text} that week, {match}."
    )

    st.markdown("#### Next to the OLS lines")
    st.markdown(
        "The other page fits a line by least squares. "
        "For a household that buys both products, Premium is 1 and Regular is 0, and the chart line is the contour where the fitted score is one half. "
        "Household 3's regression is bought versus not bought. "
        "Those contours are computed here only so the two readings can be compared. This page does not use them as constraints."
    )
    compare_rows = []
    for household in households:
        fit = fits[household["label"]]
        line = household["lines"][0]
        correct, n = classify_accuracy(frame, household, line)
        sign = "−" if fit["k"] < 0 else "+"
        given_sign = "−" if float(line["given_k"]) < 0 else "+"
        compare_rows.append({
            "Household": household["label"],
            "Given line": f"P = {float(line['given_m']):.4f}R {given_sign} {abs(float(line['given_k'])):.2f}",
            "Given": f"{correct}/{n}",
            "OLS contour": f"P = {fit['m']:.2f}R {sign} {abs(fit['k']):.2f}",
            "OLS": f"{fit['correct']}/{fit['n']}",
        })
    render_table(pd.DataFrame(compare_rows))

    threshold_household = next(
        (
            household for household in ols_detail["households"]
            if any(line.get("fixed_threshold") is not None for line in household["lines"])
        ),
        None,
    )
    if threshold_household is not None:
        level = next(
            float(line["fixed_threshold"])
            for line in threshold_household["lines"]
            if line.get("fixed_threshold") is not None
        )
        misses = _ols_misses(frame, threshold_household)
        fit = fits[threshold_household["label"]]
        if misses:
            described = "; ".join(
                f"week {miss['week']} (R = {miss['R']:.0f}, P = {miss['P']:.0f}) was {miss['observed']}, "
                f"and the OLS contour called it {miss['predicted']}"
                for miss in misses
            )
            st.markdown(
                f"{threshold_household['label']}'s OLS contour scores {fit['correct']}/{fit['n']}. "
                f"It misclassifies {described}. "
                f"The OLS page therefore uses the horizontal threshold P = {level:.2f}, "
                f"which scores {_accuracy_label_local(threshold_household)} on that page's lines."
            )
        else:
            st.markdown(
                f"{threshold_household['label']}'s OLS page uses the threshold P = {level:.2f}."
            )

    ols_shared = ols_detail["shared"]["best"]
    shared = detail["shared"]["best"]
    focus_fit = fits[focus_label]
    st.markdown(
        f"{focus_label}'s OLS slope is {focus_fit['m']:.2f}. "
        f"The given slope is {float(focus_line['given_m']):.4f}. "
        "Every week tugs the OLS line, including weeks far from the boundary, which is why that slope is steeper. "
        "The given line only has to keep the two groups apart."
    )
    st.markdown(
        f"Solved with the same quantities and the same price box, the OLS menu is "
        f"R = {money(ols_shared['R'])}, P = {money(ols_shared['P'])}, "
        f"weekly revenue {money(ols_shared['revenue'])} "
        f"({ols_shared['assignment_label']}). "
        f"This page's menu is R = {money(shared['R'])}, P = {money(shared['P'])}, "
        f"weekly revenue {money(shared['revenue'])}."
    )
    st.markdown(
        "OLS is reproducible: the same table always returns the same coefficients. "
        "A hand-placed line depends on which gap you decide to sit in. "
        "Both can separate these weeks. This page commits to the hand-placed reading."
    )


def _contour_latex(slope: float, icept: float) -> str:
    """Hand-placed lines keep four decimals on the slope and two on the intercept."""
    sign = "+" if icept >= 0 else "-"
    return rf"P = {slope:.4f}\, R {sign} {abs(icept):.2f}"


render_methodology(detail, ols_detail)
render_given_lines(detail, ols_detail)

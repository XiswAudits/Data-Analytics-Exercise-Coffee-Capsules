"""Whiteboard-style linear programme for coffee-capsule prices.

Decision variables are the Regular price R and the Premium price P. Each
household buys a fixed quantity of one product, so revenue is linear in the
two prices. Which product they buy is a straight line in the R-P plane, fitted
by OLS from the weekly data (a linear probability model). Product assignments
are separate linear programmes; the best feasible one is kept.

The earlier successive linear programme of quadratic OLS revenue has been
retired. This module is the only price model the app uses.

Class-board numbers are not copied from anywhere. Slopes, intercepts, and
quantities are estimated here. Household 2 is the exception: its switching
line is the horizontal threshold P = 77.5, which separates every week. The
OLS line misclassified week 6 (R = 55, P = 80).
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog

# Same file the Streamlit app reads. Anchored to this module so `python price_optimisation.py`
# does not depend on the process working directory.
DEFAULT_CSV = str(Path(__file__).resolve().parent / "coffee_capsules_data.csv")

FORMULATION_SUMMARY = """
Decision variables are the Regular price R and the Premium price P, both non-negative.

Each household buys a fixed quantity of one product (the average units bought on weeks they chose that product). Revenue is therefore linear: quantity times the price of the product they buy.

The product is not a decision variable inside one programme. Each household is a straight line in the R-P plane. Household 1 and Household 3 are fitted by OLS (a linear probability model at the 0.5 contour). Household 2's line is the horizontal threshold P = 77.5, because the OLS fit misclassified week 6 (R = 55, P = 80). One side of the line is Premium, the other is Regular, and a household that sometimes buys nothing also has a stop-buying line. Each assignment of products to households is its own linear programme. The assignment with the highest feasible revenue is the shared menu.

`scipy.optimize.linprog` minimises, so the objective vector is the negated quantity vector. Upper price bounds are the highest Regular and Premium prices in the experiment, so a price cannot run off to infinity. Those bounds are data, not demand coefficients.
""".strip()

LINE_STYLE = {
    "Household 1": "solid",
    "Household 2": "dash",
    "Household 3": "longdashdot",
}
LINE_COLOR = {
    "Household 1": "#172033",
    "Household 2": "#1677ff",
    "Household 3": "#ff8a1f",
}
_TOL = 1e-7
# Premium at or below this Premium price, Regular at or above it.
# Midpoint of the gap between HH2's highest Premium week (75) and lowest Regular week (80).
HH2_SWITCH_THRESHOLD = 77.5


def discover_households(frame: pd.DataFrame) -> list[str]:
    """Column prefixes that have both a Regular and a Premium quantity."""
    found = []
    for col in frame.columns:
        if not str(col).endswith("_Regular") or col == "P_Regular":
            continue
        prefix = str(col)[: -len("_Regular")]
        if prefix == "P":
            continue
        if f"{prefix}_Premium" in frame.columns:
            found.append(prefix)
    return found


def household_label(prefix: str) -> str:
    if prefix.startswith("HH") and prefix[2:].isdigit():
        return f"Household {int(prefix[2:])}"
    return prefix


def _fit_lpm(prices: np.ndarray, target: np.ndarray) -> dict:
    """OLS linear probability model. Returns intercept and slopes on R and P."""
    model = LinearRegression()
    model.fit(prices, target)
    fitted = model.predict(prices)
    resid = target - fitted
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 1e-15 else float("nan")
    return {
        "intercept": float(model.intercept_),
        "b_R": float(model.coef_[0]),
        "b_P": float(model.coef_[1]),
        "r_squared": r2,
    }


def _score(line: dict, regular: float, premium: float) -> float:
    return line["intercept"] + line["b_R"] * regular + line["b_P"] * premium


def line_equation(line: dict) -> str:
    """0.5 contour written as P = slope·R + intercept, a horizontal P, or a vertical R line."""
    return format_display_equation(line, digits=4, compact=False)


def format_display_equation(line: dict, digits: int = 2, compact: bool = True) -> str:
    """Contour for a legend or card. Two decimals read as P = 1.60R − 9.26."""
    gap = "" if compact else " "
    if line.get("given_m") is not None and digits == 2:
        # Keep the digits that identify a hand-placed line, such as 0.6667 and -0.0502.
        slope = float(line["given_m"])
        icept = float(line["given_k"])
        sign = "−" if icept < 0 else "+"
        return f"P = {slope:.4f}{gap}R {sign} {abs(icept):.2f}"
    if line.get("fixed_threshold") is not None:
        return f"P = {float(line['fixed_threshold']):.{digits}f}"
    intercept, b_r, b_p = line["intercept"], line["b_R"], line["b_P"]
    if abs(b_p) < 1e-12:
        if abs(b_r) < 1e-12:
            return "score does not depend on price"
        level = (0.5 - intercept) / b_r
        return f"R = {level:.{digits}f}"
    slope = -b_r / b_p
    icept = (0.5 - intercept) / b_p
    if abs(slope) < 1e-8:
        return f"P = {icept:.{digits}f}"
    sign = "−" if icept < 0 else "+"
    return f"P = {slope:.{digits}f}{gap}R {sign} {abs(icept):.{digits}f}"


def _halfspace(line: dict, side: str) -> tuple[np.ndarray, float]:
    """score >= 0.5 (`high`) or score <= 0.5 (`low`) as one row of A_ub x <= b."""
    intercept, b_r, b_p = line["intercept"], line["b_R"], line["b_P"]
    if side == "high":
        return np.array([-b_r, -b_p], dtype=float), intercept - 0.5
    if side == "low":
        return np.array([b_r, b_p], dtype=float), 0.5 - intercept
    raise ValueError(side)


def contour_terms(line: dict) -> tuple[float, float] | None:
    """Slope and intercept of the 0.5 contour, written P = slope·R + intercept."""
    if line.get("fixed_threshold") is not None:
        return 0.0, float(line["fixed_threshold"])
    if abs(line["b_P"]) < 1e-12:
        return None
    slope = -line["b_R"] / line["b_P"]
    icept = (0.5 - line["intercept"]) / line["b_P"]
    return float(slope), float(icept)


# Hand-placed separating lines, written P = m R + k.
# Premium (or still buying) on or below the line. Not estimated from the table.
CLASSMATE_CONTOURS = {
    "Household 1": (0.6667, 40.00, "switch"),
    "Household 2": (0.1667, 67.50, "switch"),
    "Household 3": (-0.0502, 87.38, "stop"),
}


def contour_line(slope: float, intercept: float, kind: str) -> dict:
    """Line P = slope·R + intercept, with Premium (or still buying) on or below it.

    The 0.5 contour of score = (0.5 + intercept) + slope·R − P is that line.
    score >= 0.5 is the side the linear programme treats as Premium or as still buying.
    """
    line = {
        "intercept": 0.5 + float(intercept),
        "b_R": float(slope),
        "b_P": -1.0,
        "r_squared": float("nan"),
        "kind": kind,
        "given_m": float(slope),
        "given_k": float(intercept),
    }
    line["equation"] = line_equation(line)
    return line


def classify_accuracy(frame: pd.DataFrame, household: dict, line: dict) -> tuple[int, int]:
    """Weeks whose observed product matches the product implied by `line`."""
    prefix = household["prefix"]
    probe = {"lines": [line], "only_product": household.get("only_product")}
    correct = 0
    for _, row in frame.iterrows():
        regular_qty = float(row[f"{prefix}_Regular"])
        premium_qty = float(row[f"{prefix}_Premium"])
        observed = "Regular" if regular_qty > 0 else "Premium" if premium_qty > 0 else "None"
        predicted = predict_product(probe, float(row["P_Regular"]), float(row["P_Premium"]))
        correct += int(predicted == observed)
    return correct, int(len(frame))


def ols_regression(frame: pd.DataFrame, household: dict) -> dict:
    """Least-squares line for one household, and the rows that produced it.

    A household that buys both products is a Premium/Regular regression on the
    weeks they bought something. A household that never buys Regular (Household 3)
    is a bought/not-bought regression on every week. Both are y = a + b R + c P.
    """
    prefix = household["prefix"]
    regular_qty = frame[f"{prefix}_Regular"].to_numpy(dtype=float)
    premium_qty = frame[f"{prefix}_Premium"].to_numpy(dtype=float)
    bought_regular = regular_qty > 0
    bought_premium = premium_qty > 0
    bought = bought_regular | bought_premium
    if bought_regular.any() and bought_premium.any():
        mask = bought
        target = bought_premium[mask].astype(float)
        kind = "switch"
        positive_label = "Premium"
    else:
        mask = np.ones(len(frame), dtype=bool)
        target = bought.astype(float)
        kind = "stop"
        positive_label = "bought"
    weeks = frame.loc[mask, "T"].to_numpy()
    regular = frame.loc[mask, "P_Regular"].to_numpy(dtype=float)
    premium = frame.loc[mask, "P_Premium"].to_numpy(dtype=float)
    observed = np.where(
        bought_regular[mask],
        "Regular",
        np.where(bought_premium[mask], "Premium", "None"),
    )
    line = _fit_lpm(np.column_stack([regular, premium]), target)
    line["kind"] = kind
    line["equation"] = line_equation(line)
    slope, icept = contour_terms(line)
    design = np.column_stack([np.ones(len(target)), regular, premium])
    beta = np.array([line["intercept"], line["b_R"], line["b_P"]], dtype=float)
    fitted = design @ beta
    correct, n = classify_accuracy(frame, household, line)
    rows = []
    for index, week in enumerate(weeks):
        rows.append({
            "week": int(week),
            "R": float(regular[index]),
            "P": float(premium[index]),
            "y": int(target[index]),
            "observed": str(observed[index]),
            "fitted": float(fitted[index]),
        })
    return {
        "kind": kind,
        "positive_label": positive_label,
        "rows": rows,
        "a": float(line["intercept"]),
        "b": float(line["b_R"]),
        "c": float(line["b_P"]),
        "m": float(slope),
        "k": float(icept),
        "line": line,
        "X": design.tolist(),
        "y": target.astype(float).tolist(),
        "XtX": (design.T @ design).tolist(),
        "Xty": (design.T @ target).astype(float).tolist(),
        "beta": beta.tolist(),
        "correct": correct,
        "n": n,
    }


def ols_switch_line(frame: pd.DataFrame, household: dict) -> dict | None:
    """OLS linear probability model on weeks the household bought something.

    Premium is 1 and Regular is 0. Households that replace this fit with a
    threshold still get the OLS line here, so the page can show what it missed.
    """
    prefix = household["prefix"]
    regular_qty = frame[f"{prefix}_Regular"].to_numpy(dtype=float)
    premium_qty = frame[f"{prefix}_Premium"].to_numpy(dtype=float)
    bought = (regular_qty > 0) | (premium_qty > 0)
    if int(bought.sum()) < 2 or not (regular_qty[bought] > 0).any() or not (premium_qty[bought] > 0).any():
        return None
    prices = np.column_stack([
        frame.loc[bought, "P_Regular"].to_numpy(dtype=float),
        frame.loc[bought, "P_Premium"].to_numpy(dtype=float),
    ])
    target = (premium_qty[bought] > 0).astype(float)
    line = _fit_lpm(prices, target)
    line["kind"] = "switch"
    line["equation"] = line_equation(line)
    return line


def horizontal_switch(level: float) -> dict:
    """Switching line P = `level`. Premium when P is at or below it, Regular above."""
    line = {
        "intercept": float(level) + 0.5,
        "b_R": 0.0,
        "b_P": -1.0,
        "r_squared": float("nan"),
        "kind": "switch",
        "fixed_threshold": float(level),
    }
    line["equation"] = line_equation(line)
    return line


def inequality_text(line: dict, side: str) -> str:
    """Human reading of the half-space, as a bound on P or on R."""
    if line.get("fixed_threshold") is not None:
        level = float(line["fixed_threshold"])
        wants_below = (side == "high" and line["b_P"] < 0) or (side == "low" and line["b_P"] > 0)
        op = "<=" if wants_below else ">="
        return f"P {op} {level:.4f}"
    intercept, b_r, b_p = line["intercept"], line["b_R"], line["b_P"]
    if abs(b_p) < 1e-12:
        level = (0.5 - intercept) / b_r if abs(b_r) > 1e-12 else float("nan")
        wants_low_r = (side == "high" and b_r < 0) or (side == "low" and b_r > 0)
        op = "<=" if wants_low_r else ">="
        return f"R {op} {level:.4f}"
    slope = -b_r / b_p
    icept = (0.5 - intercept) / b_p
    wants_below = (side == "high" and b_p < 0) or (side == "low" and b_p > 0)
    op = "<=" if wants_below else ">="
    sign = "+" if icept >= 0 else "−"
    return f"P {op} {slope:.4f} R {sign} {abs(icept):.4f}"


def _ols_lines(label: str, prices: np.ndarray, bought_reg: np.ndarray, bought_prem: np.ndarray) -> list[dict]:
    """OLS switching line, plus a stop line when the household sometimes buys nothing.

    Household 2's switch is the horizontal threshold instead of the OLS contour.
    """
    bought = bought_reg | bought_prem
    lines = []
    if bought.any() and (~bought).any():
        stop = _fit_lpm(prices, bought.astype(float))
        stop["kind"] = "stop"
        stop["equation"] = line_equation(stop)
        lines.append(stop)
    if int(bought.sum()) >= 2 and bought_reg[bought].any() and bought_prem[bought].any():
        if label == "Household 2":
            switch = horizontal_switch(HH2_SWITCH_THRESHOLD)
        else:
            switch = _fit_lpm(prices[bought], bought_prem[bought].astype(float))
            switch["kind"] = "switch"
            switch["equation"] = line_equation(switch)
        lines.append(switch)
    return lines


def _classmate_lines(label: str) -> list[dict]:
    """The given separating line for this household. Premium, or still buying, is on or below it."""
    if label not in CLASSMATE_CONTOURS:
        raise KeyError(f"No classmate line for {label}")
    slope, icept, kind = CLASSMATE_CONTOURS[label]
    return [contour_line(slope, icept, kind)]


def load_problem(csv_path: str = DEFAULT_CSV, line_source: str = "ols") -> dict:
    """Fixed quantities and one line per household, plus price bounds.

    `line_source="ols"` fits the lines (Household 2 uses P = 77.5).
    `line_source="classmate"` uses the hand-placed contours in `CLASSMATE_CONTOURS`.
    Quantities and bounds always come from the CSV.
    """
    if line_source not in ("ols", "classmate"):
        raise ValueError(f"Unknown line_source {line_source!r}")
    frame = pd.read_csv(csv_path)
    regular = frame["P_Regular"].to_numpy(dtype=float)
    premium = frame["P_Premium"].to_numpy(dtype=float)
    prices = np.column_stack([regular, premium])
    bounds = {
        "R_lower": 0.0,
        "P_lower": 0.0,
        "R_upper": float(np.max(regular)),
        "P_upper": float(np.max(premium)),
    }
    households = []
    for prefix in discover_households(frame):
        label = household_label(prefix)
        reg_qty = frame[f"{prefix}_Regular"].to_numpy(dtype=float)
        prem_qty = frame[f"{prefix}_Premium"].to_numpy(dtype=float)
        bought_reg = reg_qty > 0
        bought_prem = prem_qty > 0
        bought = bought_reg | bought_prem
        d_regular = float(reg_qty[bought_reg].mean()) if bought_reg.any() else 0.0
        d_premium = float(prem_qty[bought_prem].mean()) if bought_prem.any() else 0.0
        if line_source == "classmate":
            lines = _classmate_lines(label)
        else:
            lines = _ols_lines(label, prices, bought_reg, bought_prem)
        predicted = [
            predict_product({"lines": lines, "only_product": "Premium" if not bought_reg.any() else None}, r, p)
            for r, p in zip(regular, premium)
        ]
        observed = np.where(bought_reg, "Regular", np.where(bought_prem, "Premium", "None"))
        accuracy = float(np.mean([pred == obs for pred, obs in zip(predicted, observed)])) if len(observed) else float("nan")
        households.append(
            {
                "label": label,
                "prefix": prefix,
                "lines": lines,
                "d_regular": d_regular,
                "d_premium": d_premium,
                "n_regular": int(bought_reg.sum()),
                "n_premium": int(bought_prem.sum()),
                "n_none": int((~bought).sum()),
                "only_product": None if bought_reg.any() else "Premium",
                "accuracy": accuracy,
            }
        )
    return {"households": households, "bounds": bounds, "frame": frame, "line_source": line_source}


def product_choices(household: dict) -> list[str]:
    """Products this household can be assigned, given the lines the data support."""
    kinds = {line["kind"] for line in household["lines"]}
    choices = []
    if "switch" in kinds or household["d_regular"] > 0:
        choices.append("Regular")
    if household["d_premium"] > 0 or "switch" in kinds:
        choices.append("Premium")
    if "stop" in kinds:
        choices.append("None")
    if household["only_product"] == "Premium" and "Regular" in choices and "switch" not in kinds:
        choices = [c for c in choices if c != "Regular"]
    return choices


def _sides_for(household: dict, product: str) -> list[tuple[dict, str]]:
    """Behavioural half-spaces that keep `product` on the correct side of each line."""
    kinds = {line["kind"]: line for line in household["lines"]}
    sides = []
    if product == "None":
        if "stop" not in kinds:
            raise ValueError(f"{household['label']} has no stop-buying line")
        sides.append((kinds["stop"], "low"))
        return sides
    if "stop" in kinds:
        sides.append((kinds["stop"], "high"))
    if "switch" in kinds:
        sides.append((kinds["switch"], "high" if product == "Premium" else "low"))
    elif product == "Regular" and household["only_product"] == "Premium":
        raise ValueError(f"{household['label']} never buys Regular")
    return sides


def price_box_rows(bounds: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ]
    )
    rhs = np.array(
        [
            -bounds["R_lower"],
            bounds["R_upper"],
            -bounds["P_lower"],
            bounds["P_upper"],
        ],
        dtype=float,
    )
    names = [
        f"R >= {bounds['R_lower']:.4f}",
        f"R <= {bounds['R_upper']:.4f}",
        f"P >= {bounds['P_lower']:.4f}",
        f"P <= {bounds['P_upper']:.4f}",
    ]
    return rows, rhs, names


def build_constraint_system(households: list[dict], assignment: dict[str, str], bounds: dict) -> dict:
    """Stack behavioural half-spaces and the price box into A_ub x <= b_ub."""
    rows = []
    rhs = []
    names = []
    for household in households:
        product = assignment[household["label"]]
        for line, side in _sides_for(household, product):
            coeff, limit = _halfspace(line, side)
            rows.append(coeff)
            rhs.append(limit)
            kind = "switches so that" if line["kind"] == "switch" else "keeps buying so that"
            if product == "None":
                kind = "stops buying so that"
            names.append(f"{household['label']} {kind} {inequality_text(line, side)}")
    box_rows, box_rhs, box_names = price_box_rows(bounds)
    if rows:
        matrix = np.vstack([np.vstack(rows), box_rows])
        limits = np.concatenate([np.array(rhs, dtype=float), box_rhs])
    else:
        matrix = box_rows
        limits = box_rhs
    return {"A_ub": matrix, "b_ub": limits, "names": names + box_names}


def objective_coefficients(households: list[dict], assignment: dict[str, str]) -> np.ndarray:
    """c such that revenue = c · [R, P]. None contributes nothing."""
    c_r = 0.0
    c_p = 0.0
    for household in households:
        product = assignment[household["label"]]
        if product == "Regular":
            c_r += household["d_regular"]
        elif product == "Premium":
            c_p += household["d_premium"]
    return np.array([c_r, c_p], dtype=float)


def _vertices_from_constraints(matrix: np.ndarray, limits: np.ndarray) -> list[np.ndarray]:
    points = []
    n = len(limits)
    for i in range(n):
        for j in range(i + 1, n):
            pair = np.vstack([matrix[i], matrix[j]])
            if abs(np.linalg.det(pair)) < 1e-10:
                continue
            try:
                point = np.linalg.solve(pair, np.array([limits[i], limits[j]], dtype=float))
            except np.linalg.LinAlgError:
                continue
            if np.all(matrix @ point <= limits + 1e-6):
                points.append(point)
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - kept) < 1e-6 for kept in unique):
            unique.append(point)
    if not unique:
        return []
    centroid = np.mean(unique, axis=0)
    unique.sort(key=lambda pt: np.arctan2(pt[1] - centroid[1], pt[0] - centroid[0]))
    return unique


def _tight_constraints(matrix: np.ndarray, limits: np.ndarray, point: np.ndarray) -> list[int]:
    slack = limits - matrix @ point
    return [i for i, gap in enumerate(slack) if abs(gap) <= 1e-5]


def _is_vertex(matrix: np.ndarray, limits: np.ndarray, point: np.ndarray) -> bool:
    tight = _tight_constraints(matrix, limits, point)
    for i in tight:
        for j in tight:
            if j <= i:
                continue
            if abs(np.linalg.det(np.vstack([matrix[i], matrix[j]]))) > 1e-8:
                return True
    return False


def _prefer_matching_vertex(
    households: list[dict],
    assignment: dict[str, str],
    coeff: np.ndarray,
    point: np.ndarray,
    revenue: float,
    vertices: list[np.ndarray],
) -> np.ndarray:
    """Keep a flat-edge optimum on the side of the line that matches its product.

    P = 77.5 is Premium (P <= 77.5) and also meets a Regular constraint (P >= 77.5).
    If HiGHS sits on that boundary for a Regular assignment, report another optimal
    vertex where the line says Regular, such as P = 100.
    """

    def matches(pt: np.ndarray) -> bool:
        return all(
            predict_product(household, float(pt[0]), float(pt[1])) == assignment[household["label"]]
            for household in households
        )

    if matches(point):
        return point
    candidates = [vertex for vertex in vertices if abs(float(coeff @ vertex) - revenue) <= 1e-4 and matches(vertex)]
    if not candidates:
        return point
    return max(candidates, key=lambda vertex: (float(vertex[1]), float(vertex[0])))


def solve_assignment(households: list[dict], assignment: dict[str, str], bounds: dict) -> dict:
    """One linear programme: this product assignment, maximised with linprog."""
    system = build_constraint_system(households, assignment, bounds)
    coeff = objective_coefficients(households, assignment)
    # linprog minimises. Negating the quantity vector maximises revenue.
    result = linprog(
        c=-coeff,
        A_ub=system["A_ub"],
        b_ub=system["b_ub"],
        bounds=[(bounds["R_lower"], bounds["R_upper"]), (bounds["P_lower"], bounds["P_upper"])],
        method="highs",
    )
    label = ", ".join(f"{name} buys {product}" for name, product in assignment.items())
    solved = {
        "assignment": dict(assignment),
        "assignment_label": label,
        "c": coeff,
        "A_ub": system["A_ub"],
        "b_ub": system["b_ub"],
        "constraint_names": system["names"],
        "status": int(result.status),
        "status_name": result.message,
        "feasible": bool(result.success),
    }
    if not result.success:
        solved.update(
            {
                "R": float("nan"),
                "P": float("nan"),
                "revenue": float("-inf"),
                "vertex": False,
                "tight": [],
                "satisfied": False,
            }
        )
        return solved
    point = np.asarray(result.x, dtype=float)
    revenue = float(coeff @ point)
    vertices = _vertices_from_constraints(system["A_ub"], system["b_ub"])
    point = _prefer_matching_vertex(households, assignment, coeff, point, revenue, vertices)
    revenue = float(coeff @ point)
    tight_idx = _tight_constraints(system["A_ub"], system["b_ub"], point)
    solved.update(
        {
            "R": float(point[0]),
            "P": float(point[1]),
            "revenue": revenue,
            "vertex": _is_vertex(system["A_ub"], system["b_ub"], point),
            "tight": [system["names"][i] for i in tight_idx],
            "satisfied": bool(np.all(system["A_ub"] @ point <= system["b_ub"] + 1e-6)),
            "vertices": vertices,
            "flat_price": (
                "Premium price is not in this objective, so revenue is unchanged along that edge; a vertex of that edge is reported."
                if abs(coeff[1]) < 1e-12
                else "Regular price is not in this objective, so revenue is unchanged along that edge; a vertex of that edge is reported."
                if abs(coeff[0]) < 1e-12
                else ""
            ),
        }
    )
    return solved


def enumerate_assignments(households: list[dict]) -> list[dict[str, str]]:
    """Cartesian product of each household's feasible products."""
    menus: list[dict[str, str]] = [{}]
    for household in households:
        extended = []
        for product in product_choices(household):
            for menu in menus:
                nxt = dict(menu)
                nxt[household["label"]] = product
                extended.append(nxt)
        menus = extended
    return menus


def best_programme(households: list[dict], bounds: dict, assignments: list[dict[str, str]] | None = None) -> dict:
    """Solve every assignment and keep the highest feasible revenue."""
    if assignments is None:
        assignments = enumerate_assignments(households)
    solved = [solve_assignment(households, menu, bounds) for menu in assignments]
    feasible = [item for item in solved if item["feasible"] and np.isfinite(item["revenue"])]
    if not feasible:
        return {"best": None, "cases": solved}
    best = max(feasible, key=lambda item: item["revenue"])
    return {"best": best, "cases": solved}


def predict_product(household: dict, regular: float, premium: float) -> str:
    """Product implied by which side of the fitted lines the price sits on."""
    kinds = {line["kind"]: line for line in household["lines"]}
    if "stop" in kinds and _score(kinds["stop"], regular, premium) < 0.5 - 1e-9:
        return "None"
    if "switch" in kinds:
        return "Premium" if _score(kinds["switch"], regular, premium) >= 0.5 - 1e-9 else "Regular"
    return household.get("only_product") or "Premium"


def revenue_of(household: dict, regular: float, premium: float) -> dict:
    product = predict_product(household, regular, premium)
    if product == "Regular":
        revenue = household["d_regular"] * regular
    elif product == "Premium":
        revenue = household["d_premium"] * premium
    else:
        revenue = 0.0
    return {"product": product, "revenue": float(revenue)}


def _format_objective(coeff: np.ndarray) -> str:
    parts = []
    if abs(coeff[0]) > 1e-12:
        parts.append(f"{coeff[0]:.4f} R")
    if abs(coeff[1]) > 1e-12:
        parts.append(f"{coeff[1]:.4f} P")
    if not parts:
        return "max  0"
    return "max  " + " + ".join(parts)


def format_programme(solved: dict, households: list[dict] | None = None) -> str:
    """Objective and every constraint, with the fitted numbers filled in."""
    lines = [
        "Decision variables: R (Regular price), P (Premium price).",
        _format_objective(solved["c"]),
        "subject to",
    ]
    for name in solved["constraint_names"]:
        lines.append(f"  {name}")
    if households is not None:
        lines.append("assignment")
        for label, product in solved["assignment"].items():
            match = next(item for item in households if item["label"] == label)
            quantity = {"Regular": match["d_regular"], "Premium": match["d_premium"], "None": 0.0}[product]
            lines.append(f"  {label} buys {product} (d = {quantity:.4f})")
    if solved["feasible"]:
        lines.append(
            f"optimum  R = {solved['R']:.4f}, P = {solved['P']:.4f}, revenue = {solved['revenue']:.4f}"
        )
        lines.append("binding constraints")
        for name in solved["tight"]:
            lines.append(f"  {name}")
    else:
        lines.append("infeasible")
    return "\n".join(lines)


def household_equations(household: dict) -> list[str]:
    """One readable equation per fitted line, plus the fixed quantities."""
    text = []
    for line in household["lines"]:
        if line.get("fixed_threshold") is not None:
            level = float(line["fixed_threshold"])
            text.append(
                f"P = {level:.4f}: Premium when P <= {level:.4f}, Regular when P >= {level:.4f}"
            )
            continue
        if line["kind"] == "switch":
            role = "switches between Regular and Premium on"
        else:
            role = "stops buying above"
        text.append(f"{role} {line['equation']}")
    text.append(
        f"d_regular = {household['d_regular']:.4f} (n = {household['n_regular']}), "
        f"d_premium = {household['d_premium']:.4f} (n = {household['n_premium']})"
    )
    return text


def optimise_prices_detailed(csv_path: str = DEFAULT_CSV, line_source: str = "ols") -> dict:
    """Shared menu plus one programme per household, and every assignment tried.

    `line_source` defaults to the OLS lines, which is what the root app shows.
    """
    problem = load_problem(csv_path, line_source=line_source)
    households = problem["households"]
    bounds = problem["bounds"]
    shared = best_programme(households, bounds)
    per_household = []
    for household in households:
        solo = best_programme([household], bounds)
        per_household.append({"household": household, "programme": solo})
    return {
        "households": households,
        "bounds": bounds,
        "frame": problem["frame"],
        "shared": shared,
        "per_household": per_household,
        "formulation": format_programme(shared["best"], households) if shared["best"] else "No feasible shared programme.",
        "line_source": line_source,
    }


def _shock_households(households: list[dict], label: str, field: str, factor: float) -> list[dict]:
    shocked = []
    for household in households:
        clone = {
            **household,
            "lines": [dict(line) for line in household["lines"]],
        }
        if clone["label"] == label:
            for line in clone["lines"]:
                if "fixed_threshold" in line or field not in line:
                    continue
                line[field] = line[field] * factor
                line["equation"] = line_equation(line)
        shocked.append(clone)
    return shocked


def _shock_threshold(households: list[dict], label: str, level: float) -> list[dict]:
    """Replace one household's horizontal threshold and leave every other line alone."""
    shocked = []
    for household in households:
        clone = {
            **household,
            "lines": [dict(line) for line in household["lines"]],
        }
        if clone["label"] == label:
            clone["lines"] = [
                horizontal_switch(level) if "fixed_threshold" in line else line
                for line in clone["lines"]
            ]
        shocked.append(clone)
    return shocked


def _sensitivity_row(household_label: str, parameter: str, shock: str, best: dict | None, base_revenue: float) -> dict:
    revenue = best["revenue"] if best else float("nan")
    return {
        "Household": household_label,
        "kind": "line coefficient",
        "parameter": parameter,
        "shock": shock,
        "R_opt": best["R"] if best else float("nan"),
        "P_opt": best["P"] if best else float("nan"),
        "max_revenue": revenue,
        "delta_revenue": revenue - base_revenue if best else float("nan"),
        "assignment": best["assignment_label"] if best else "infeasible",
        "status": "optimal" if best else "infeasible",
    }


def _shock_given_contour(households: list[dict], label: str, which: str, factor: float) -> list[dict]:
    """Move the slope m or the intercept k of one household's given line."""
    shocked = []
    for household in households:
        lines = []
        for line in household["lines"]:
            if household["label"] == label and "given_m" in line:
                slope = float(line["given_m"]) * (factor if which == "m" else 1.0)
                icept = float(line["given_k"]) * (factor if which == "k" else 1.0)
                lines.append(contour_line(slope, icept, line["kind"]))
            else:
                lines.append(dict(line))
        shocked.append({**household, "lines": lines})
    return shocked


OUTSIDE_PRICE_STATUS = "outside tested price range"
OUTSIDE_PRICE_LABEL = "outside price range"


def _inside_price_box(regular: float, premium: float, bounds: dict) -> bool:
    """True when both prices sit inside the tested box, including the edges."""
    return (
        bounds["R_lower"] - _TOL <= regular <= bounds["R_upper"] + _TOL
        and bounds["P_lower"] - _TOL <= premium <= bounds["P_upper"] + _TOL
    )


def structural_sensitivity(detail: dict, pct: float = 0.10) -> pd.DataFrame:
    """Re-solve after shocking each line, and after moving prices.

    OLS lines are shocked on intercept and on each slope. A fixed threshold,
    such as Household 2's P = 77.5, is shocked by moving that level ±pct.
    A hand-placed contour is shocked on its slope m and its intercept k.
    A price move that leaves the tested box is not scored; its status is
    "outside tested price range".
    """
    rows = []
    base = detail["shared"]["best"]
    bounds = detail["bounds"]
    fields = (("intercept", "intercept"), ("b_R", "slope on R"), ("b_P", "slope on P"))
    for household in detail["households"]:
        fixed = [line for line in household["lines"] if "fixed_threshold" in line]
        given = [line for line in household["lines"] if "given_m" in line]
        fitted = [
            line for line in household["lines"]
            if "fixed_threshold" not in line and "given_m" not in line
        ]
        if given:
            for which, pretty in (("m", "slope m"), ("k", "intercept k")):
                for sign, factor in (("+", 1.0 + pct), ("−", 1.0 - pct)):
                    shocked = _shock_given_contour(detail["households"], household["label"], which, factor)
                    solved = best_programme(shocked, bounds)
                    rows.append(_sensitivity_row(household["label"], pretty, f"{sign}{pct:.0%}", solved["best"], base["revenue"]))
        if fitted:
            for field, pretty in fields:
                for sign, factor in (("+", 1.0 + pct), ("−", 1.0 - pct)):
                    shocked = _shock_households(detail["households"], household["label"], field, factor)
                    solved = best_programme(shocked, bounds)
                    rows.append(_sensitivity_row(household["label"], pretty, f"{sign}{pct:.0%}", solved["best"], base["revenue"]))
        for line in fixed:
            level = float(line["fixed_threshold"])
            for sign, factor in (("+", 1.0 + pct), ("−", 1.0 - pct)):
                shocked = _shock_threshold(detail["households"], household["label"], level * factor)
                solved = best_programme(shocked, bounds)
                rows.append(
                    _sensitivity_row(
                        household["label"],
                        f"threshold P = {level:.4f}",
                        f"{sign}{pct:.0%}",
                        solved["best"],
                        base["revenue"],
                    )
                )
        own = next(item["programme"]["best"] for item in detail["per_household"] if item["household"]["label"] == household["label"])
        for price_name, index in (("R", 0), ("P", 1)):
            for sign, factor in (("+", 1.0 + pct), ("−", 1.0 - pct)):
                point = [own["R"], own["P"]]
                point[index] = point[index] * factor
                row = {
                    "Household": household["label"],
                    "kind": "price move",
                    "parameter": price_name,
                    "shock": f"{sign}{pct:.0%}",
                    "R_opt": point[0],
                    "P_opt": point[1],
                }
                if not _inside_price_box(point[0], point[1], bounds):
                    row.update(
                        max_revenue=float("nan"),
                        delta_revenue=float("nan"),
                        assignment="",
                        status=OUTSIDE_PRICE_STATUS,
                    )
                else:
                    assessed = revenue_of(household, point[0], point[1])
                    row.update(
                        max_revenue=assessed["revenue"],
                        delta_revenue=assessed["revenue"] - own["revenue"],
                        assignment=assessed["product"],
                        status="evaluated",
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def regret_table(detail: dict, scenario: tuple[float, float]) -> pd.DataFrame:
    """Revenue and regret of the shared optimum, each household optimum, and the sliders."""
    shared = detail["shared"]["best"]
    menus = [("Shared LP optimum", shared["R"], shared["P"])]
    for item in detail["per_household"]:
        best = item["programme"]["best"]
        menus.append((f"{item['household']['label']} LP optimum", best["R"], best["P"]))
    menus.append(("Scenario prices", float(scenario[0]), float(scenario[1])))
    rows = []
    for item in detail["per_household"]:
        household = item["household"]
        own = item["programme"]["best"]["revenue"]
        for menu, regular, premium in menus:
            assessed = revenue_of(household, regular, premium)
            rows.append(
                {
                    "Household": household["label"],
                    "Menu": menu,
                    "R": regular,
                    "P": premium,
                    "product": assessed["product"],
                    "revenue": assessed["revenue"],
                    "regret": own - assessed["revenue"],
                }
            )
    return pd.DataFrame(rows)


_CHART_FONT = "Inter, Arial, sans-serif"


def apply_chart_layout(
    fig: go.Figure,
    *,
    height: int,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    show_legend: bool = True,
    left: int = 72,
    right: int = 28,
    bottom: int | None = None,
    top: int | None = None,
) -> go.Figure:
    """One layout for every chart: title clear of the plot, legend in a row underneath."""
    named = [trace for trace in fig.data if getattr(trace, "name", None) and trace.showlegend is not False]
    legend_rows = 1 if len(named) <= 3 else 2 if len(named) <= 6 else 3
    if bottom is None:
        bottom = (28 + 26 * legend_rows + (18 if xaxis_title else 0)) if show_legend else (48 if xaxis_title else 36)
    if top is None:
        top = 56 if title else 16
    fig.update_layout(
        font=dict(family=_CHART_FONT, size=13, color="#172033"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        margin=dict(l=left, r=right, t=top, b=bottom, autoexpand=True),
        hoverlabel=dict(font=dict(family=_CHART_FONT, size=12), bgcolor="white", bordercolor="#e5eaf0"),
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            xref="container",
            y=1,
            yanchor="top",
            yref="container",
            pad=dict(t=10, b=2, l=2, r=0),
            font=dict(family=_CHART_FONT, size=15, color="#101828"),
        ) if title else None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.012,
            yref="container",
            xanchor="center",
            x=0.5,
            xref="container",
            font=dict(family=_CHART_FONT, size=12, color="#344054"),
            bgcolor="rgba(255,255,255,0)",
            itemsizing="constant",
            traceorder="normal",
        ),
        showlegend=show_legend,
    )
    fig.update_xaxes(
        automargin=True,
        tickfont=dict(family=_CHART_FONT, size=12, color="#475467"),
        title=dict(
            text=xaxis_title or "",
            standoff=10,
            font=dict(family=_CHART_FONT, size=13, color="#344054"),
        ),
    )
    fig.update_yaxes(
        automargin=True,
        tickfont=dict(family=_CHART_FONT, size=12, color="#475467"),
        title=dict(
            text=yaxis_title or "",
            standoff=12,
            font=dict(family=_CHART_FONT, size=13, color="#344054"),
        ),
    )
    return fig


def _clip_segment(
    x0: float, y0: float, x1: float, y1: float, xr: list[float], yr: list[float]
) -> tuple[float, float, float, float] | None:
    """Liang–Barsky clip of a segment to the axis rectangle."""
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xr[0]), (dx, xr[1] - x0), (-dy, y0 - yr[0]), (dy, yr[1] - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    if t0 > t1:
        return None
    return (x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy)


def _axis_window(
    solved: dict,
    bounds: dict,
    extra: list[tuple[float, float]] | None = None,
) -> tuple[list[float], list[float]]:
    """Range that contains the whole feasible region, with room for the optimum label."""
    pts = [np.asarray(pt, dtype=float) for pt in (solved.get("vertices") or [])]
    if solved.get("feasible"):
        pts.append(np.array([float(solved["R"]), float(solved["P"])], dtype=float))
    for regular, premium in extra or []:
        pts.append(np.array([float(regular), float(premium)], dtype=float))
    if len(pts) < 1:
        return [bounds["R_lower"] - 4.0, bounds["R_upper"] + 6.0], [bounds["P_lower"] - 4.0, bounds["P_upper"] + 8.0]
    arr = np.vstack(pts)
    r0, r1 = float(arr[:, 0].min()), float(arr[:, 0].max())
    p0, p1 = float(arr[:, 1].min()), float(arr[:, 1].max())
    span_r = max(r1 - r0, 8.0)
    span_p = max(p1 - p0, 8.0)
    pad_r = max(8.0, min(12.0, 0.55 * span_r))
    pad_p = max(10.0, min(14.0, 0.6 * span_p))
    # A little context outside the price box, without a deep empty margin.
    x0 = max(bounds["R_lower"] - 4.0, r0 - pad_r)
    y0 = max(bounds["P_lower"] - 4.0, p0 - pad_p)
    return [x0, r1 + pad_r], [y0, p1 + pad_p]


def _add_household_lines(fig: go.Figure, households: list[dict], span: tuple[float, float]) -> None:
    r_grid = np.linspace(span[0], span[1], 120)
    for household in households:
        for line in household["lines"]:
            if abs(line["b_P"]) < 1e-12:
                continue
            slope = -line["b_R"] / line["b_P"]
            icept = (0.5 - line["intercept"]) / line["b_P"]
            p_grid = slope * r_grid + icept
            fig.add_trace(
                go.Scatter(
                    x=r_grid,
                    y=p_grid,
                    mode="lines",
                    name=f"{household['label']}: {format_display_equation(line)}",
                    line=dict(
                        color=LINE_COLOR.get(household["label"], "#172033"),
                        dash=LINE_STYLE.get(household["label"], "solid"),
                        width=2.5,
                    ),
                    hovertemplate="R=%{x:.2f}<br>P=%{y:.2f}<extra>" + household["label"] + "</extra>",
                )
            )


def _shade(fig: go.Figure, solved: dict) -> None:
    vertices = solved.get("vertices") or []
    if len(vertices) < 3:
        return
    xs = [pt[0] for pt in vertices] + [vertices[0][0]]
    ys = [pt[1] for pt in vertices] + [vertices[0][1]]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            fill="toself",
            mode="lines",
            name="Feasible region",
            line=dict(color="rgba(22,119,255,0.35)", width=1),
            fillcolor="rgba(22,119,255,0.16)",
            hoverinfo="skip",
        )
    )


def _objective_level(fig: go.Figure, solved: dict, x_range: list[float], y_range: list[float]) -> None:
    coeff = solved["c"]
    if not solved["feasible"]:
        return
    revenue = solved["revenue"]
    if abs(coeff[1]) > 1e-10:
        y_at = lambda regular: (revenue - coeff[0] * regular) / coeff[1]
        y0, y1 = y_at(x_range[0]), y_at(x_range[1])
        # Pull a nearby iso-revenue line fully inside the axes instead of clipping it.
        if min(y0, y1) >= y_range[0] - 16 and max(y0, y1) <= y_range[1] + 16:
            y_range[0] = min(y_range[0], min(y0, y1) - 2.0)
            y_range[1] = max(y_range[1], max(y0, y1) + 2.0)
        clipped = _clip_segment(x_range[0], y0, x_range[1], y1, x_range, y_range)
        if clipped is None:
            return
        xa, ya, xb, yb = clipped
        fig.add_trace(
            go.Scatter(
                x=[xa, xb],
                y=[ya, yb],
                mode="lines",
                name="Objective level",
                line=dict(color="#0f9f6e", width=2, dash="dot"),
                hovertemplate="Iso-revenue<extra></extra>",
            )
        )
    elif abs(coeff[0]) > 1e-10 and x_range[0] <= solved["R"] <= x_range[1]:
        fig.add_trace(
            go.Scatter(
                x=[solved["R"], solved["R"]],
                y=[y_range[0], y_range[1]],
                mode="lines",
                name="Objective level",
                line=dict(color="#0f9f6e", width=2, dash="dot"),
                hovertemplate="Iso-revenue<extra></extra>",
            )
        )


def _optimum_marker(fig: go.Figure, solved: dict, x_range: list[float], y_range: list[float]) -> None:
    if not solved["feasible"]:
        return
    fig.add_trace(
        go.Scatter(
            x=[solved["R"]],
            y=[solved["P"]],
            mode="markers",
            name="Optimum",
            marker=dict(size=14, color="#172033", symbol="star", line=dict(color="white", width=1)),
            hovertemplate=f"R=%{{x:.2f}}<br>P=%{{y:.2f}}<br>Revenue={solved['revenue']:.2f}<extra></extra>",
        )
    )
    span_x = x_range[1] - x_range[0]
    span_y = y_range[1] - y_range[0]
    near_right = solved["R"] >= x_range[0] + 0.5 * span_x
    near_top = solved["P"] >= y_range[0] + 0.58 * span_y
    text_x = solved["R"] - 0.045 * span_x if near_right else solved["R"] + 0.045 * span_x
    text_y = solved["P"] - 0.07 * span_y if near_top else solved["P"] + 0.07 * span_y
    text_x = min(max(text_x, x_range[0] + 0.22 * span_x), x_range[1] - 0.04 * span_x)
    text_y = min(max(text_y, y_range[0] + 0.16 * span_y), y_range[1] - 0.08 * span_y)
    fig.add_annotation(
        x=solved["R"],
        y=solved["P"],
        ax=text_x,
        ay=text_y,
        axref="x",
        ayref="y",
        xanchor="right" if text_x <= solved["R"] else "left",
        yanchor="top" if text_y <= solved["P"] else "bottom",
        text=(
            f"P = {solved['P']:.2f}<br>R = {solved['R']:.2f}<br>Revenue = {solved['revenue']:.2f}"
        ),
        showarrow=True,
        arrowhead=2,
        arrowwidth=1.2,
        arrowcolor="#172033",
        bgcolor="white",
        bordercolor="#d0d5dd",
        borderwidth=1,
        borderpad=5,
        align="left",
        font=dict(family=_CHART_FONT, size=12, color="#172033"),
    )


def _price_axes(fig: go.Figure, bounds: dict, x_range: list[float], y_range: list[float], height: int = 560) -> None:
    fig.update_xaxes(range=list(x_range), zeroline=False, gridcolor="#e8edf3", showgrid=True)
    fig.update_yaxes(range=list(y_range), zeroline=False, gridcolor="#e8edf3", showgrid=True)
    apply_chart_layout(
        fig,
        height=height,
        xaxis_title="Regular price R",
        yaxis_title="Premium price P",
        left=78,
        right=36,
        bottom=108,
        top=20,
    )
    fig.add_shape(
        type="rect",
        x0=bounds["R_lower"],
        x1=bounds["R_upper"],
        y0=bounds["P_lower"],
        y1=bounds["P_upper"],
        line=dict(color="rgba(23,32,51,0.28)", width=1, dash="dot"),
        fillcolor="rgba(0,0,0,0)",
    )


def _price_figure(detail: dict, solved: dict, households: list[dict], frame: pd.DataFrame | None = None) -> go.Figure:
    """Feasible region, lines, and optimum, framed so labels stay inside the axes."""
    bounds = detail["bounds"]
    extra = None
    if frame is not None:
        extra = list(
            zip(
                frame["P_Regular"].to_numpy(dtype=float),
                frame["P_Premium"].to_numpy(dtype=float),
            )
        )
    x_range, y_range = _axis_window(solved, bounds, extra)
    fig = go.Figure()
    _shade(fig, solved)
    _add_household_lines(fig, households, (x_range[0], x_range[1]))
    if frame is not None and len(households) == 1:
        _add_observations(fig, frame, households[0])
    _objective_level(fig, solved, x_range, y_range)
    fig.update_xaxes(range=list(x_range))
    fig.update_yaxes(range=list(y_range))
    _optimum_marker(fig, solved, x_range, y_range)
    _price_axes(fig, bounds, x_range, y_range, height=580 if frame is None else 620)
    return fig


def shared_figure(detail: dict) -> go.Figure:
    """All three household lines, the shared feasible region, and one optimum."""
    return _price_figure(detail, detail["shared"]["best"], detail["households"])


def _add_observations(fig: go.Figure, frame: pd.DataFrame, household: dict) -> None:
    """Weekly choices for one household: blue Regular, orange Premium."""
    prefix = household["prefix"]
    regular_price = frame["P_Regular"].to_numpy(dtype=float)
    premium_price = frame["P_Premium"].to_numpy(dtype=float)
    regular_qty = frame[f"{prefix}_Regular"].to_numpy(dtype=float)
    premium_qty = frame[f"{prefix}_Premium"].to_numpy(dtype=float)
    weeks = frame["T"].to_numpy()
    choice = np.where(regular_qty > 0, "Regular", np.where(premium_qty > 0, "Premium", "No purchase"))
    quantity = np.where(regular_qty > 0, regular_qty, premium_qty)
    colors = {"Regular": "#1677ff", "Premium": "#ff8a1f", "No purchase": "#98a2b3"}
    for name, color in colors.items():
        mask = choice == name
        if not np.any(mask):
            continue
        fig.add_trace(
            go.Scatter(
                x=regular_price[mask],
                y=premium_price[mask],
                mode="markers",
                name=f"Observed {name}",
                marker=dict(size=11, color=color, line=dict(color="white", width=1.4)),
                customdata=np.column_stack([weeks[mask], quantity[mask]]),
                hovertemplate=(
                    "Week %{customdata[0]:.0f}<br>R=%{x:.0f}<br>P=%{y:.0f}"
                    "<br>quantity %{customdata[1]:.0f}<extra>" + name + "</extra>"
                ),
            )
        )


def household_figure(detail: dict, label: str) -> go.Figure:
    """One household: its line, observed weeks, feasible region, and optimum."""
    item = next(entry for entry in detail["per_household"] if entry["household"]["label"] == label)
    return _price_figure(detail, item["programme"]["best"], [item["household"]], frame=detail["frame"])


def _shock_tick(parameter: str, shock: str) -> str:
    text = str(parameter)
    if "threshold" in text and "=" in text:
        level = float(text.split("=")[-1])
        text = f"P = {level:.2f}"
    text = text.replace("slope on R", "slope R").replace("slope on P", "slope P")
    return f"{text} {shock}"


def _bar_chart(table: pd.DataFrame, title: str, yaxis_title: str) -> go.Figure:
    fig = go.Figure()
    for household, color in LINE_COLOR.items():
        part = table[table["Household"] == household]
        if part.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[_shock_tick(row.parameter, row.shock) for row in part.itertuples()],
                y=part["delta_revenue"],
                name=household,
                marker_color=color,
                hovertemplate="%{x}<br>%{y:.2f} €<extra>" + household + "</extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-28)
    apply_chart_layout(
        fig,
        height=460,
        title=title,
        yaxis_title=yaxis_title,
        left=78,
        right=24,
        top=58,
        bottom=120,
    )
    fig.update_yaxes(zeroline=True, zerolinecolor="#d0d5dd", gridcolor="#e8edf3")
    return fig


def structural_sensitivity_figure(table: pd.DataFrame) -> go.Figure:
    coef = table[table["kind"] == "line coefficient"]
    return _bar_chart(coef, "Shared revenue when a line moves ±10%", "Change in shared revenue (€)")


def _grouped_bar_center_offsets(n_series: int, bargap: float = 0.2) -> list[float]:
    """Distance from the category index to each grouped bar's center.

    Plotly's group layout uses this when bargroupgap is 0: the gap between
    groups is `bargap`, and the bars in a group split the rest evenly.
    """
    if n_series <= 0:
        return []
    slot = (1.0 - bargap) / n_series
    return [(2 * index + 1 - n_series) * slot / 2 for index in range(n_series)]


def price_move_figure(table: pd.DataFrame) -> go.Figure:
    moves = table[table["kind"] == "price move"].copy()
    outside = (
        moves["status"].eq(OUTSIDE_PRICE_STATUS)
        if "status" in moves.columns
        else pd.Series(False, index=moves.index)
    )
    plotted = moves.copy()
    plotted.loc[outside, "delta_revenue"] = float("nan")
    fig = _bar_chart(plotted, "Revenue when that household's price moves ±10%", "Change versus own optimum (€)")
    # Pin the gap so the annotation sits on the empty bar's slot.
    fig.update_layout(bargap=0.2, bargroupgap=0)
    present = [name for name in LINE_COLOR if (moves["Household"] == name).any()]
    if not present or not outside.any():
        return fig
    offsets = _grouped_bar_center_offsets(len(present))
    order = [
        _shock_tick(row.parameter, row.shock)
        for row in moves[moves["Household"] == present[0]].itertuples()
    ]
    category_index = {label: index for index, label in enumerate(order)}
    for row in moves.loc[outside].itertuples():
        tick = _shock_tick(row.parameter, row.shock)
        series = present.index(row.Household)
        fig.add_annotation(
            x=category_index[tick] + offsets[series],
            y=0,
            text=OUTSIDE_PRICE_LABEL,
            showarrow=False,
            font=dict(family=_CHART_FONT, size=10, color="#98a2b3"),
            textangle=-90,
            xanchor="center",
            yanchor="top",
            yshift=-6,
        )
    return fig


def _num(value: float) -> str:
    return f"{value:,.2f}"


def _signed(delta: float) -> str:
    if abs(delta) < 0.005:
        return "0.00"
    if delta < 0:
        return f"−{abs(delta):,.2f}"
    return f"+{delta:,.2f}"


def _line_level(line: dict, regular: float) -> float | None:
    terms = contour_terms(line)
    if terms is None:
        return None
    slope, icept = terms
    return slope * regular + icept


def _paid_price(product: str) -> str:
    if product == "Regular":
        return "R"
    if product == "Premium":
        return "P"
    return ""


def _purchase_calc(household: dict, product: str, regular: float, premium: float, revenue: float) -> str:
    if product == "Regular":
        return f"{_num(household['d_regular'])} × {_num(regular)} = {_num(revenue)}"
    if product == "Premium":
        return f"{_num(household['d_premium'])} × {_num(premium)} = {_num(revenue)}"
    return _num(revenue)


def _level_move(old_level: float | None, new_level: float | None) -> str:
    if old_level is None or new_level is None:
        return "moves"
    if new_level < old_level - 0.005:
        return "drops"
    if new_level > old_level + 0.005:
        return "rises"
    return "stays"


def _move_bullet(household: dict, own: dict, row) -> str:
    """One plain line for a computed price move. Figures come from that row."""
    name = f"{row.parameter} {row.shock}"
    regular = float(row.R_opt)
    premium = float(row.P_opt)
    if str(row.status) == OUTSIDE_PRICE_STATUS:
        return f"{name}: Outside tested price range: not calculated"
    product = str(row.assignment)
    own_product = own["assignment"][household["label"]]
    revenue = float(row.max_revenue)
    delta = float(row.delta_revenue)
    own_r = float(own["R"])
    own_p = float(own["P"])
    prices_same = abs(regular - own_r) < 0.005 and abs(premium - own_p) < 0.005
    if prices_same:
        return f"{name}: No change: prices stay R {_num(regular)}, P {_num(premium)}"
    paid = _paid_price(product)
    if abs(delta) < 0.005 and product == own_product and paid and paid != str(row.parameter):
        return f"{name}: No change: it doesn't buy that product"
    kinds = {line["kind"]: line for line in household["lines"]}
    calc = _purchase_calc(household, product, regular, premium, revenue)
    if product == "None" and own_product != "None":
        if str(row.parameter) == "P":
            return f"{name}: Stops buying: P passes its stop line, loses all {_num(own['revenue'])}"
        line = kinds.get("stop")
        new_level = _line_level(line, regular) if line else None
        old_level = _line_level(line, own_r) if line else None
        if new_level is None:
            return f"{name}: Stops buying: loses all {_num(own['revenue'])}"
        return (
            f"{name}: at R = {_num(regular)} its line {_level_move(old_level, new_level)} to {_num(new_level)}, "
            f"so {own_product} at {_num(own_p)} is above it; it stops buying, loses all {_num(own['revenue'])}"
        )
    if product != own_product and product in ("Regular", "Premium"):
        line = kinds.get("switch") or kinds.get("stop")
        new_level = _line_level(line, regular) if line else None
        old_level = _line_level(line, own_r) if line else None
        if str(row.parameter) == "R" and new_level is not None:
            relation = "above" if own_p > new_level + 1e-9 else "below"
            return (
                f"{name}: at R = {_num(regular)} its line {_level_move(old_level, new_level)} to {_num(new_level)}, "
                f"so {own_product} at {_num(own_p)} is {relation} it; "
                f"it switches to {product} ({calc}), {_signed(delta)}"
            )
        if new_level is not None:
            relation = "above" if premium > new_level + 1e-9 else "below"
            return (
                f"{name}: P moves to {_num(premium)}, {relation} its line at {_num(new_level)}; "
                f"it switches to {product} ({calc}), {_signed(delta)}"
            )
        return f"{name}: it switches to {product} ({calc}), {_signed(delta)}"
    return f"{name}: still buys {product} ({calc}), {_signed(delta)}"


def price_move_overview(detail: dict, moves: pd.DataFrame) -> str:
    """Compact card under the price-move chart. Every figure comes from the solved rows."""
    if moves.empty:
        return ""
    columns = []
    for label in moves["Household"].drop_duplicates():
        household = next(item for item in detail["households"] if item["label"] == label)
        own = next(
            item["programme"]["best"]
            for item in detail["per_household"]
            if item["household"]["label"] == label
        )
        product = own["assignment"][label]
        bullets = "".join(
            f"<li>{escape(_move_bullet(household, own, row))}</li>"
            for row in moves[moves["Household"] == label].itertuples(index=False)
        )
        columns.append(
            '<div class="overview-house">'
            f'<div class="overview-kicker">{escape(str(label))}</div>'
            f'<p class="overview-best">Best: {escape(product)} at R {_num(own["R"])}, '
            f'P {_num(own["P"])}, revenue {_num(own["revenue"])}.</p>'
            f'<ul class="overview-moves">{bullets}</ul>'
            "</div>"
        )
    takeaway = (
        "Cutting a price earns less from its buyers; raising it past a household's line "
        "makes it switch or stop buying, which costs the most."
    )
    return (
        '<div class="card overview-card">'
        '<div class="card-title">Results overview</div>'
        f'<div class="overview-grid">{"".join(columns)}</div>'
        f'<p class="overview-takeaway"><strong>Takeaway.</strong> {escape(takeaway)}</p>'
        "</div>"
    )


_MENU_LABEL = {
    "Shared LP optimum": "Shared LP",
    "Household 1 LP optimum": "HH1 LP",
    "Household 2 LP optimum": "HH2 LP",
    "Household 3 LP optimum": "HH3 LP",
    "Scenario prices": "Scenario",
}


def regret_figure(table: pd.DataFrame) -> go.Figure:
    pivot = table.pivot(index="Household", columns="Menu", values="regret")
    columns = [_MENU_LABEL.get(str(name), str(name)) for name in pivot.columns]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=columns,
            y=list(pivot.index),
            colorscale="Blues",
            colorbar=dict(
                title=dict(text="Regret (€)", font=dict(family=_CHART_FONT, size=12, color="#344054")),
                thickness=14,
                len=0.7,
                tickfont=dict(family=_CHART_FONT, size=11),
            ),
            text=np.round(pivot.to_numpy(), 2),
            texttemplate="%{text:.2f}",
            textfont=dict(family=_CHART_FONT, size=12, color="#172033"),
            hovertemplate="%{y}<br>%{x}<br>regret €%{z:.2f}<extra></extra>",
            xgap=4,
            ygap=4,
        )
    )
    apply_chart_layout(
        fig,
        height=380,
        title="Regret versus each household's own optimum",
        show_legend=False,
        left=120,
        right=88,
        top=58,
        bottom=56,
    )
    return fig


def _accuracy_label(household: dict) -> str:
    n = household["n_regular"] + household["n_premium"] + household["n_none"]
    if n == 0 or not np.isfinite(household["accuracy"]):
        return ""
    correct = int(round(household["accuracy"] * n))
    return f"{correct}/{n}"


def constraint_table(solved: dict, households: list[dict]) -> pd.DataFrame:
    """Objective, both sides of each line, and the price box, with binding flags.

    A side that is not part of this assignment is listed so the unused inequality
    stays visible. Binding is only meaningful for rows that are in the programme.
    """
    if solved is None:
        return pd.DataFrame(columns=["Piece", "Constraint", "Role", "Binding", "In-sample accuracy"])
    rows = [{
        "Piece": "Objective",
        "Constraint": _format_objective(solved["c"]),
        "Role": "maximise revenue",
        "Binding": "",
        "In-sample accuracy": "",
    }]
    active = set(solved["constraint_names"])
    tight = set(solved.get("tight") or [])
    for household in households:
        accuracy = _accuracy_label(household)
        for line in household["lines"]:
            if line["kind"] == "switch":
                options = (
                    ("Premium", "high", "switches so that"),
                    ("Regular", "low", "switches so that"),
                )
            else:
                options = (
                    ("still buys", "high", "keeps buying so that"),
                    ("stops buying", "low", "stops buying so that"),
                )
            for role, side, kind in options:
                text = inequality_text(line, side)
                full = f"{household['label']} {kind} {text}"
                if full in tight:
                    binding = "yes"
                elif full in active:
                    binding = "no"
                else:
                    binding = "not in this programme"
                rows.append({
                    "Piece": household["label"],
                    "Constraint": text,
                    "Role": role,
                    "Binding": binding,
                    "In-sample accuracy": accuracy,
                })
    for name in solved["constraint_names"]:
        if name.startswith("R ") or name.startswith("P "):
            rows.append({
                "Piece": "Price bound",
                "Constraint": name,
                "Role": "box",
                "Binding": "yes" if name in tight else "no",
                "In-sample accuracy": "",
            })
    return pd.DataFrame(rows)


def sanity_check(detail: dict) -> list[str]:
    """Optimum on a vertex, constraints satisfied, revenue equals c·x."""
    problems = []
    shared = detail["shared"]["best"]
    if shared is None:
        return ["shared programme infeasible"]
    point = np.array([shared["R"], shared["P"]])
    if not shared["satisfied"]:
        problems.append("shared optimum violates A_ub x <= b_ub")
    if not shared["vertex"]:
        problems.append("shared optimum is not a vertex")
    if abs(float(shared["c"] @ point) - shared["revenue"]) > 1e-6:
        problems.append("shared revenue is not c·x")
    bounds = detail["bounds"]
    if not (bounds["R_lower"] - 1e-8 <= shared["R"] <= bounds["R_upper"] + 1e-8):
        problems.append("shared R outside bounds")
    if not (bounds["P_lower"] - 1e-8 <= shared["P"] <= bounds["P_upper"] + 1e-8):
        problems.append("shared P outside bounds")
    best_revenue = shared["revenue"]
    for case in detail["shared"]["cases"]:
        if case["feasible"] and case["revenue"] > best_revenue + 1e-6:
            problems.append(f"a better assignment was discarded: {case['assignment_label']}")
    for item in detail["per_household"]:
        solo = item["programme"]["best"]
        if solo is None or not solo["vertex"] or not solo["satisfied"]:
            problems.append(f"{item['household']['label']} optimum failed the vertex check")
    return problems


def _print_report(detail: dict) -> None:
    print("Per-household lines and fixed quantities")
    for household in detail["households"]:
        print(household["label"])
        for line in household_equations(household):
            print(" ", line)
        print(f"  in-sample line accuracy {household['accuracy']:.3f}")
    print()
    print("Shared linear programme")
    print(detail["formulation"])
    print()
    print("Assignments evaluated")
    for case in sorted(detail["shared"]["cases"], key=lambda item: item["revenue"], reverse=True):
        flag = "feasible" if case["feasible"] else "infeasible"
        revenue = f"{case['revenue']:.4f}" if case["feasible"] else "—"
        print(f"  {case['assignment_label']}: {flag}, revenue {revenue}")
    print()
    print("Per-household optima")
    for item in detail["per_household"]:
        best = item["programme"]["best"]
        print(
            f"  {item['household']['label']}: {best['assignment_label']}, "
            f"R={best['R']:.4f}, P={best['P']:.4f}, revenue={best['revenue']:.4f}, vertex={best['vertex']}"
        )
    problems = sanity_check(detail)
    print()
    print("Sanity:", "ok" if not problems else "; ".join(problems))


if __name__ == "__main__":
    _print_report(optimise_prices_detailed())

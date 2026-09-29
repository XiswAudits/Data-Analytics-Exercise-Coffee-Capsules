# Coffee Capsule Pricing Model

Interactive Streamlit app for the 11-week, three-household coffee-capsule exercise. The published model is the whiteboard linear programme from [Coffee-Capsules-Experiment](https://github.com/XiswAudits/Coffee-Capsules-Experiment) (`e1abc10`).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

`analysis/pricing_optimization/app.py` is a compatibility entry point for the existing Streamlit Community Cloud path. It runs the same root `app.py`.

The price calculation can also be checked without the dashboard:

```bash
python price_optimisation.py
```

## Revenue-maximising prices

The question is which Premium price `P` and Regular price `R` maximise profit. Capsule costs are not in the data, so the programme maximises **revenue**. The only decision variables are the two prices. Each household buys a fixed quantity of the product it chooses, so revenue is linear in those prices.

The calculation lives in `price_optimisation.py` and is shown in the root `app.py` under **Revenue-maximising prices** (tabs **Shared prices** and **Per household**).

The earlier successive linear programme of quadratic OLS demand has been retired. The app now has this one model.

### Fixed quantities

`d_i` is the average number of capsules bought on weeks when that household chose the product. Estimated from `coffee_capsules_data.csv`, not hard-coded:

| Household | d when buying Regular | d when buying Premium |
|---|---:|---:|
| Household 1 | 4.3333 (9 weeks) | 4.0000 (2 weeks) |
| Household 2 | 4.0000 (7 weeks) | 3.0000 (4 weeks) |
| Household 3 | never buys Regular | 4.2857 (7 weeks) |

### Lines in the R–P plane

A linear probability model (OLS of a 0/1 indicator on an intercept, `R`, and `P`) supplies the 0.5 contour for Household 1 and Household 3. Premium is the side of a switching line where the fitted score is at least 0.5. A household that sometimes buys nothing also has a stop-buying line.

Household 2's OLS line (`P = 0.2305 R + 67.6909`) classified 10 of 11 weeks and put week 6 (R = 55, P = 80) on the Premium side. It is replaced by the horizontal threshold `P = 77.5`, which separates all 11 weeks: Premium weeks 1, 4, 5 and 10 have P from 70 to 75, and Regular weeks have P at least 80. Household 2 buys Premium when `P <= 77.5` and Regular when `P >= 77.5`.

| Household | Line | Reading |
|---|---|---|
| Household 1 | `P = 1.6030 R − 9.2621` | Always buys. Premium at or below the line, Regular above it. In-sample accuracy 11/11. |
| Household 2 | `P = 77.5` | Always buys. Premium at or below the threshold, Regular at or above it. In-sample accuracy 11/11. |
| Household 3 | `P = 0.1205 R + 79.2906` | Never buys Regular. Still buys Premium at or below the line, and stops above it. Accuracy 11/11. |

### Shared linear programme

Every assignment of products is solved as its own LP with `scipy.optimize.linprog` (HiGHS). The solver minimises, so the objective vector is the **negated** quantity vector. The best feasible assignment on this sample is Household 1 Premium, Household 2 Regular, Household 3 Premium:

```text
max  4.0000 R + 8.2857 P
subject to
  Household 1 buys Premium:  P <= 1.6030 R − 9.2621
  Household 2 buys Regular:  P >= 77.5
  Household 3 still buys:    P <= 0.1205 R + 79.2906
  0 <= R <= 65
  0 <= P <= 100
```

The upper bounds are the highest Regular and Premium prices in the experiment, so a price cannot run to infinity. Optimum: **R = €65.00**, **P = €87.12**, **revenue = €981.86**. It is a vertex: `R <= 65` and Household 3's buying line are both binding, and the constraints hold.

Other feasible assignments earn less (about €875, €868, €799, €640, and €542). Two assignments are infeasible inside the price box.

### Per-household programmes

Each household is also solved alone, on its own line and its own quantity:

| Household | Buys | Optimal R | Optimal P | Revenue |
|---|---|---:|---:|---:|
| Household 1 | Premium | €65.00 | €94.94 | €379.74 |
| Household 2 | Regular | €65.00 | €100.00 | €260.00 |
| Household 3 | Premium | €65.00 | €87.12 | €373.38 |

Household 2's revenue does not depend on `P` once they are kept on the Regular side, so every feasible Premium price on that edge earns €260. HiGHS returns the vertex `P = 100`.

`python price_optimisation.py` checks that each reported optimum is a vertex, satisfies `A_ub x <= b_ub`, and that revenue equals `c · x`.

The page opens on **Revenue-maximising prices**. A **Model & methodology** section at the bottom explains the data, the fixed quantities, the lines, the linear programme, the result, the checks, and the limitations. KPI cards show the shared optimum. A household control switches between all households and one household: all households draw every line and the shared feasible region; one household draws its line, the observed weeks, and its own optimum. The constraints table lists the objective, both sides of each line, the price bounds, whether each row binds, and in-sample accuracy. Average weekly demand is the fixed quantity `d_i`. Sensitivity re-solves the shared LP after a ±10% shock to each OLS line coefficient and to Household 2's threshold, and moves each household's own prices by ±10%. The regret matrix includes the shared optimum, each household optimum, and the scenario prices.

## Dataset

`coffee_capsules_data.csv` is the file the app and `price_optimisation.py` read. It is the experiment schema (`T`, `P_Regular`, `P_Premium`, `HH1_Regular`, …). `data/coffee_capsules.csv` is this repository's earlier transcription of the same 11 weeks (same quantities and prices, different column names) and is kept because the observations match. Zero quantities are retained because they represent observed No Purchase behaviour.

## Limitations

Only 11 weekly observations are available. The fitted lines and the linear programme are exploratory. They are not causal elasticities, willingness-to-pay estimates, or out-of-sample forecasts. No unit costs were supplied, so the objective is revenue, not profit.

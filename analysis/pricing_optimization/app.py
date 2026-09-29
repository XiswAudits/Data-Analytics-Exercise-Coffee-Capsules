import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.optimize import minimize

st.set_page_config(page_title="Darbs | Pricing Lab", page_icon="◉", layout="wide")
st.markdown("""
<style>
.stApp{background:#f5f5f7;color:#1d1d1f}
.block-container{padding-top:2rem;max-width:1400px}
[data-testid="stMetric"]{background:white;border:1px solid #e8e8ed;padding:18px;border-radius:18px}
section[data-testid="stSidebar"]{background:#fff}
h1,h2,h3{letter-spacing:-.04em}
div[data-testid="stExpander"]{background:white;border-radius:14px;border:1px solid #e8e8ed}
</style>
""",unsafe_allow_html=True)

@st.cache_data
def data():
 return pd.DataFrame({"Week":range(1,12),"Regular_Price":[40,35,50,65,50,55,40,45,47,35,45],"Premium_Price":[70,81,100,75,70,80,90,92,91,70,80],
 "HH1_Regular":[5,5,4,0,0,4,4,4,4,5,4],"HH1_Premium":[0,0,0,4,4,0,0,0,0,0,0],
 "HH2_Regular":[0,4,4,0,0,4,4,4,4,0,4],"HH2_Premium":[3,0,0,3,3,0,0,0,0,3,0],
 "HH3_Regular":[0]*11,"HH3_Premium":[4,4,0,4,5,4,0,0,0,5,4]})
df=data()
households=["HH1","HH2","HH3"]
with st.sidebar:
 st.markdown("## ◉ Darbs")
 st.caption("PRICING INTELLIGENCE")
 page=st.radio("Workspace",["Overview","Household explorer","Optimization","Sensitivity & regret","Dataset"])
 st.divider()
 rlo,rhi=st.slider("Regular price bounds",0,150,(20,100))
 plo,phi=st.slider("Premium price bounds",0,180,(30,120))
 st.caption("Model uses all 11 weeks, including zero purchases.")

@st.cache_data
def fit_models(frame):
 out={}
 X=sm.add_constant(frame[["Regular_Price","Premium_Price"]])
 for h in households:
  for prod in ["Regular","Premium"]:
   res=sm.OLS(frame[f"{h}_{prod}"],X).fit()
   b=res.params.copy()
   own="Regular_Price" if prod=="Regular" else "Premium_Price"
   b[own]=min(b[own],0)
   out[(h,prod)]=(res,b)
 return out
models=fit_models(df)
def qs(r,p,scale=None):
 total={"Regular":0.,"Premium":0.}
 for (h,prod),(res,b) in models.items():
  fac=1 if scale is None else scale
  q=max(0,float(np.array([1,r,p])@b.values)*fac)
  total[prod]+=q
 return total
def rev(r,p,scale=None):
 q=qs(r,p,scale)
 return r*q["Regular"]+p*q["Premium"]
def opt(scale=1):
 z=minimize(lambda x:-rev(x[0],x[1],scale),[(rlo+rhi)/2,(plo+phi)/2],method="SLSQP",bounds=[(rlo,rhi),(plo,phi)])
 return z.x[0],z.x[1],-z.fun
ro,po,rv=opt()
def header(title,sub):
 st.title(title);st.caption(sub)
def note(txt):
 st.markdown(f"<div style='color:#6e6e73;font-size:14px;margin-top:-8px;margin-bottom:18px'>{txt}</div>",unsafe_allow_html=True)

if page=="Overview":
 header("Pricing intelligence","A demand-led view of capsule pricing across three household segments.")
 a,b,c=st.columns(3)
 a.metric("Optimized revenue",f"{rv:,.2f}")
 b.metric("Regular price",f"{ro:.2f}")
 c.metric("Premium price",f"{po:.2f}")
 note("The model fits separate OLS demand equations for each household and capsule, then searches the selected price bounds for the revenue-maximizing pair.")
 st.subheader("Demand by household")
 rows=[]
 for h in households:
  for prod in ["Regular","Premium"]:
   res,coef=models[(h,prod)]
   rows.append({"Household":h,"Capsule":prod,"Mean weekly units":df[f"{h}_{prod}"].mean(),"R²":res.rsquared})
 fig=px.bar(pd.DataFrame(rows),x="Household",y="Mean weekly units",color="Capsule",barmode="group",template="plotly_white")
 st.plotly_chart(fig,use_container_width=True)
 st.subheader("Model workflow")
 st.markdown("**1 · Estimate** household-specific OLS → **2 · Shape** own-price slopes to non-positive → **3 · Optimize** total revenue → **4 · Stress-test** coefficient uncertainty → **5 · Compare** strategies using regret.")
elif page=="Household explorer":
 header("Household explorer","Select a household to inspect its observed demand, fitted equations, and price response.")
 h=st.selectbox("Household",households)
 prod=st.segmented_control("Capsule",["Regular","Premium"],default="Regular")
 res,beta=models[(h,prod)]
 st.subheader(f"{h} · {prod} demand")
 st.latex(r"Q=\beta_0+\beta_R R+\beta_P P")
 st.code(f"Q = {beta.iloc[0]:.4f} {beta.iloc[1]:+.4f}·R {beta.iloc[2]:+.4f}·P")
 st.caption(f"OLS fit: R² = {res.rsquared:.3f}. Own-price slope is constrained to be non-positive for the pricing model; equation coefficients shown are the adjusted coefficients.")
 col1,col2=st.columns(2)
 col1.metric("Mean weekly purchases",f"{df[f'{h}_{prod}'].mean():.2f}")
 col2.metric("OLS R²",f"{res.rsquared:.3f}")
 d=df[["Week","Regular_Price","Premium_Price",f"{h}_{prod}"]].rename(columns={f"{h}_{prod}":"Observed units"})
 st.plotly_chart(px.line(d,x="Week",y="Observed units",markers=True,template="plotly_white",title="Observed weekly purchases (zeros retained)"),use_container_width=True)
 st.subheader("Interactive price response")
 r=st.slider("Regular price R",rlo,rhi,int(np.clip(50,rlo,rhi)))
 p=st.slider("Premium price P",plo,phi,int(np.clip(80,plo,phi)))
 pred=max(0,float(np.array([1,r,p])@beta.values))
 st.metric("Predicted weekly units",f"{pred:.2f}")
 st.caption("Change either price to see the fitted household-level quantity response. Predictions are clipped at zero.")
elif page=="Optimization":
 header("Revenue optimization","Explore the feasible price rectangle and the model's revenue surface.")
 rgrid=np.linspace(rlo,rhi,55);pgrid=np.linspace(plo,phi,55)
 z=np.array([[rev(r,p) for r in rgrid] for p in pgrid])
 fig=go.Figure(go.Contour(x=rgrid,y=pgrid,z=z,colorscale="Viridis",contours={"showlabels":True}))
 fig.add_trace(go.Scatter(x=[ro],y=[po],mode="markers+text",text=["Optimum"],textposition="top center",marker={"size":14,"color":"red"}))
 fig.update_layout(template="plotly_white",xaxis_title="Regular price R",yaxis_title="Premium price P")
 st.plotly_chart(fig,use_container_width=True)
 note("Each contour connects price combinations with equal modeled revenue. The star marks the optimizer's solution inside the user-selected bounds.")
 st.latex(r"\max_{R,P}\; R\sum_i Q_{i,R}(R,P)+P\sum_i Q_{i,P}(R,P)")
 st.write(f"**Solution:** R = {ro:.2f}, P = {po:.2f}; modeled revenue = {rv:,.2f}.")
elif page=="Sensitivity & regret":
 header("Sensitivity & regret","Test demand-scale states and compare pricing strategies without hiding trade-offs.")
 states={"Low":.9,"Base":1.,"High":1.1}
 strategies={"ODS":(ro,po),"BS1":(60,80),"BS2":(40,100)}
 payoff=pd.DataFrame({s:{k:rev(*v,scale) for k,v in strategies.items()} for s,scale in states.items()})
 regret=payoff.copy()
 for s in states: regret[s]=payoff[s].max()-payoff[s]
 st.subheader("Payoff table · revenue")
 st.dataframe(payoff.style.format("{:,.2f}"),use_container_width=True)
 st.caption("Low/base/high represent uniform 10% down, unchanged, or 10% up demand scaling—not statistical confidence intervals.")
 st.subheader("Regret matrix")
 st.dataframe(regret.style.format("{:,.2f}"),use_container_width=True)
 maxreg=regret.max(axis=1)
 st.metric("Minimum maximum regret",f"{maxreg.min():,.2f}",help=f"Strategy: {maxreg.idxmin()}")
 st.bar_chart(maxreg)
 st.subheader("Coefficient perturbation")
 n=st.slider("Simulation runs",20,500,100,20)
 rng=np.random.default_rng(7); vals=[]
 for j in range(n):
  # Demand scale shocks provide an interpretable robust-price distribution.
  scale=rng.uniform(.9,1.1)
  rr,pp,vv=opt(scale)
  vals.append({"Run":j+1,"R*":rr,"P*":pp,"Revenue":vv})
 sim=pd.DataFrame(vals)
 st.plotly_chart(px.scatter(sim,x="R*",y="P*",color="Revenue",template="plotly_white",title="Optimal prices across demand-scale scenarios"),use_container_width=True)
 st.caption("This interactive simulation perturbs demand level uniformly. For coefficient-by-coefficient perturbations, use the companion Python analysis script.")
elif page=="Dataset":
 header("Source data","The complete panel is shown, including every zero-purchase week.")
 st.dataframe(df,use_container_width=True)
 st.caption("Weekly household quantities and the two observed capsule prices. All 11 observations are included in averages and regressions.")

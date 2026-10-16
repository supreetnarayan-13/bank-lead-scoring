import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Lead Scoring", layout="wide")

@st.cache_data
def load_data():
    d = pd.read_csv("data/bank-full.csv", sep=";")
    d["converted"] = (d["y"] == "yes").astype(int)
    return d

df = load_data()
model = joblib.load("model/model.joblib")
gt = pd.read_csv("model/gain_table.csv")

page = st.sidebar.radio("Go to", ["Dashboard", "Call planner", "Score a lead", "Ask"])

# ---------- Dashboard ----------
if page == "Dashboard":
    st.title("Conversion dashboard")
    col = st.selectbox("Group by", ["job", "month", "poutcome", "education", "marital"])
    rate = df.groupby(col)["converted"].mean().sort_values(ascending=False)
    st.bar_chart(rate)
    st.caption("Share of customers who said yes in each group.")

# ---------- Call planner ----------
elif page == "Call planner":
    st.title("Call planner")
    cost = st.slider("Cost per call", 1, 50, 5)
    profit = st.slider("Profit per sale", 10, 500, 100)

    g = gt.copy()
    g["profit"] = g["y"].cumsum() * profit - (g.index + 1) * cost
    best = g["profit"].idxmax()

    c1, c2, c3 = st.columns(3)
    c1.metric("Best number of calls", f"{best + 1:,}")
    c2.metric("Share of customers", f"{g.loc[best, 'pct_called']:.0%}")
    c3.metric("Expected profit", f"{g.loc[best, 'profit']:,.0f}")
    st.line_chart(g.set_index("pct_called")["profit"])
    st.caption("Cost and profit are adjustable assumptions, not real bank numbers.")

# ---------- Score a lead ----------
elif page == "Score a lead":
    st.title("Score one customer")
    row = pd.read_csv("model/sample_row.csv")

    row["age"] = st.number_input("Age", 18, 95, 40)
    row["job"] = st.selectbox("Job", sorted(df["job"].unique()))
    row["marital"] = st.selectbox("Marital", sorted(df["marital"].unique()))
    row["education"] = st.selectbox("Education", sorted(df["education"].unique()))
    row["default"] = st.selectbox("Credit default?", ["no", "yes"])
    row["balance"] = st.number_input("Balance", -5000, 100000, 1000)
    row["housing"] = st.selectbox("Housing loan?", ["yes", "no"])
    row["loan"] = st.selectbox("Personal loan?", ["no", "yes"])
    row["contact"] = st.selectbox("Contact type", sorted(df["contact"].unique()))
    row["month"] = st.selectbox("Month", sorted(df["month"].unique()))
    row["poutcome"] = st.selectbox("Previous outcome", sorted(df["poutcome"].unique()))

    if st.button("Get score"):
        p = model.predict_proba(row)[0, 1]
        st.metric("Chance of saying yes", f"{p:.1%}")

# ---------- Ask ----------
elif page == "Ask":
    import anthropic

    st.title("Ask about the data")
    question = st.text_input("Example: Which job groups convert best?")

    if question:
        # Python does the maths. Claude only explains it, so numbers aren't invented.
        facts = {
            "overall_conversion": round(df["converted"].mean(), 4),
            "by_job": df.groupby("job")["converted"].mean().round(3).to_dict(),
            "by_month": df.groupby("month")["converted"].mean().round(3).to_dict(),
            "by_poutcome": df.groupby("poutcome")["converted"].mean().round(3).to_dict(),
            "by_education": df.groupby("education")["converted"].mean().round(3).to_dict(),
            "by_marital": df.groupby("marital")["converted"].mean().round(3).to_dict(),
        }

        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        reply = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=500,
            system=(
                "You answer questions about a bank telemarketing dataset. "
                "Use ONLY the numbers in the facts provided. "
                "If the facts do not contain the answer, say so. Keep answers short."
            ),
            messages=[{"role": "user", "content": f"Facts: {facts}\n\nQuestion: {question}"}],
        )
        st.write(reply.content[0].text)
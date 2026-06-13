import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ================= SETTINGS =================
st.set_page_config(
    page_title="AI Customer Intelligence Platform",
    layout="wide"
)

# ================= LOAD DATA =================
df = pd.read_csv("customer_dataset.csv")

# ================= LOAD MODEL =================
try:
    model = joblib.load("model.pkl")
except:
    model = None

# ================= TITLE =================
st.title("📊 AI-Driven Customer Intelligence Platform")

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard",
    "Prediction",
    "Customer Segmentation",
    "Churn Analysis",
    "Recommendation Engine"
])

# ==================================================
# DASHBOARD
# ==================================================
with tab1:

    st.header("📈 Business Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", len(df))
    c2.metric("Average Budget", f"₹{df['monthly_spend'].mean():,.0f}")
    c3.metric("Total Revenue", f"₹{df['monthly_spend'].sum():,.0f}")
    c4.metric("Churn Rate", f"{df['churn'].mean()*100:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.pie(
            df,
            names="segment",
            title="Customer Segment Distribution"
        )
        st.plotly_chart(fig1, width="stretch", key="dashboard_pie")

    with col2:
        fig2 = px.histogram(
            df,
            x="monthly_spend",
            title="Monthly Spend Distribution"
        )
        st.plotly_chart(fig2, width="stretch", key="dashboard_hist")

    st.subheader("📋 Customer Dataset")
    st.dataframe(df, width="stretch")

# ==================================================
# PREDICTION
# ==================================================
with tab2:

    st.header("🔮 Customer Segment Prediction")

    age = st.number_input("Age", 18, 100, 25)
    monthly_spend = st.number_input("Monthly Spend", 0, 100000, 30000)
    tenure = st.number_input("Tenure Months", 0, 120, 12)
    support_tickets = st.number_input("Support Tickets", 0, 50, 2)

    if st.button("Predict Segment"):

        if model is not None:

            input_data = pd.DataFrame([[
                age,
                monthly_spend,
                tenure,
                support_tickets,
                0
            ]], columns=[
                "age",
                "monthly_spend",
                "tenure_months",
                "support_tickets",
                "segment_encoded"
            ])

            prediction = model.predict(input_data)[0]

        else:
            prediction = "Customer"

        st.success(f"🎯 Predicted Segment: {prediction}")

        st.subheader("📋 Customer Details")

        result_df = pd.DataFrame({
            "Feature": [
                "Age",
                "Monthly Spend",
                "Tenure Months",
                "Support Tickets"
            ],
            "Value": [
                age,
                monthly_spend,
                tenure,
                support_tickets
            ]
        })

        st.dataframe(result_df, width="stretch")

        st.subheader("📊 Customer Profile Graph")

        graph_df = pd.DataFrame({
            "Feature": [
                "Age",
                "Monthly Spend",
                "Tenure",
                "Tickets"
            ],
            "Value": [
                age,
                monthly_spend,
                tenure,
                support_tickets
            ]
        })

        fig3 = px.bar(
            graph_df,
            x="Feature",
            y="Value",
            title="Customer Profile"
        )

        st.plotly_chart(fig3, width="stretch", key="prediction_bar")

        st.subheader("🥧 Input Distribution")

        fig4 = px.pie(
            graph_df,
            names="Feature",
            values="Value"
        )

        st.plotly_chart(fig4, width="stretch", key="prediction_pie")

        st.subheader("📥 Download Dataset")

        st.download_button(
            "Download Dataset",
            df.to_csv(index=False),
            "customer_dataset.csv",
            "text/csv"
        )

# ==================================================
# CUSTOMER SEGMENTATION
# ==================================================
with tab3:

    st.header("👥 Customer Segmentation")

    fig5 = px.pie(
        df,
        names="segment",
        title="Customer Segment Distribution"
    )

    st.plotly_chart(fig5, width="stretch", key="segment_pie")

    fig6 = px.scatter(
        df,
        x="age",
        y="monthly_spend",
        color="segment",
        title="Age vs Monthly Spend"
    )

    st.plotly_chart(fig6, width="stretch", key="segment_scatter")

    st.subheader("📋 Segment Statistics")

    segment_stats = df.groupby("segment").agg({
        "age": "mean",
        "monthly_spend": "mean"
    })

    st.dataframe(segment_stats, width="stretch")

    st.subheader("📄 Dataset")

    st.dataframe(df.head(20), width="stretch")

    st.download_button(
        "Download Dataset",
        df.to_csv(index=False),
        "customer_dataset.csv",
        "text/csv",
        key="segment_download"
    )

# ==================================================
# CHURN ANALYSIS
# ==================================================

with tab4:


  st.header("📉 Customer Churn Analysis")

  # KPIs
  col1, col2, col3 = st.columns(3)

  churn_customers = len(df[df["churn"] == 1])
  active_customers = len(df[df["churn"] == 0])
  churn_rate = round(df["churn"].mean() * 100, 2)

col1.metric("Churn Customers", churn_customers)
col2.metric("Active Customers", active_customers)
col3.metric("Churn Rate", f"{churn_rate}%")

st.markdown("---")

# Graphs
col1, col2 = st.columns(2)

with col1:

    fig1 = px.pie(
        df,
        names="churn",
        title="Customer Churn Distribution"
    )

    st.plotly_chart(
        fig1,
        width="stretch",
        key="churn_pie"
    )

with col2:

    churn_count = df["churn"].value_counts().reset_index()
    churn_count.columns = ["Status", "Count"]

    fig2 = px.bar(
        churn_count,
        x="Status",
        y="Count",
        title="Churn Count"
    )

    st.plotly_chart(
        fig2,
        width="stretch",
        key="churn_bar"
    )

st.markdown("---")

# Churn Customers Table
st.subheader("📋 Churn Customers Table")

churn_df = df[df["churn"] == 1]

st.dataframe(
    churn_df,
    width="stretch"
)

# Churn Statistics
st.subheader("📊 Churn Statistics")

stats = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Churn Customers",
        "Active Customers",
        "Churn Rate (%)"
    ],
    "Value": [
        len(df),
        churn_customers,
        active_customers,
        churn_rate
    ]
})

st.table(stats)

st.markdown("---")

# Churn Prediction
st.subheader("🔮 Churn Prediction")

age = st.number_input(
    "Age",
    18,
    100,
    30,
    key="churn_age"
)

spend = st.number_input(
    "Monthly Spend",
    0,
    100000,
    30000,
    key="churn_spend"
)

tenure = st.number_input(
    "Tenure Months",
    0,
    120,
    12,
    key="churn_tenure"
)

tickets = st.number_input(
    "Support Tickets",
    0,
    50,
    2,
    key="churn_tickets"
)

if st.button("Predict Churn Risk"):

    if tickets > 5 or tenure < 6:
        st.error("⚠ High Churn Risk")
    else:
        st.success("✅ Low Churn Risk")

st.markdown("---")

# Download Dataset
st.subheader("📥 Download Dataset")

st.download_button(
    label="Download Churn Dataset",
    data=churn_df.to_csv(index=False),
    file_name="churn_dataset.csv",
    mime="text/csv",
    key="churn_download"
)


# ==================================================
# RECOMMENDATION ENGINE
# ==================================================
with tab5:

    st.header("🤖 Smart Recommendation Engine")

    st.subheader("Select Customer Preferences")

    budget = st.selectbox(
        "Budget Range",
        ["Low", "Medium", "High"]
    )

    category = st.selectbox(
        "Preferred Category",
        ["Electronics", "Fashion", "Home", "Books"]
    )

    # Product Recommendations
    recommendations = {
        ("Low", "Electronics"): [
            "Wireless Mouse",
            "USB Keyboard",
            "Bluetooth Speaker"
        ],
        ("Medium", "Electronics"): [
            "Smart Watch",
            "Headphones",
            "Tablet"
        ],
        ("High", "Electronics"): [
            "Gaming Laptop",
            "iPhone",
            "Premium Camera"
        ],
        ("Low", "Fashion"): [
            "T-Shirt",
            "Jeans",
            "Sneakers"
        ],
        ("Medium", "Fashion"): [
            "Jacket",
            "Watch",
            "Formal Shoes"
        ],
        ("High", "Fashion"): [
            "Designer Suit",
            "Luxury Watch",
            "Premium Shoes"
        ],
        ("Low", "Home"): [
            "LED Bulb",
            "Storage Box",
            "Water Bottle"
        ],
        ("Medium", "Home"): [
            "Mixer",
            "Microwave",
            "Chair"
        ],
        ("High", "Home"): [
            "Smart TV",
            "Refrigerator",
            "Sofa Set"
        ],
        ("Low", "Books"): [
            "Python Basics",
            "Data Science Guide",
            "AI Handbook"
        ],
        ("Medium", "Books"): [
            "Machine Learning",
            "Deep Learning",
            "SQL Guide"
        ],
        ("High", "Books"): [
            "AI Research Collection",
            "Advanced ML",
            "Big Data Analytics"
        ]
    }

    products = recommendations[(budget, category)]

    st.subheader("🎯 Recommended Products")

    rec_df = pd.DataFrame({
        "Product": products
    })

    st.dataframe(rec_df, width="stretch")

    # Recommendation Graph
    st.subheader("📊 Recommendation Graph")

    graph_df = pd.DataFrame({
        "Product": products,
        "Score": [95, 90, 85]
    })

    fig = px.bar(
        graph_df,
        x="Product",
        y="Score",
        title="Recommendation Scores"
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="recommendation_graph"
    )

    # Customer Dataset
    st.subheader("📋 Customer Dataset")

    st.dataframe(df.head(20), width="stretch")

    # Download Dataset
    st.subheader("📥 Download Dataset")

    st.download_button(
        label="Download Customer Dataset",
        data=df.to_csv(index=False),
        file_name="customer_dataset.csv",
        mime="text/csv"
    )

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Customer AI Dashboard")

st.title("AI Driven Customer Analysis System")

df = pd.read_csv("customer_dataset.csv")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard",
    "Prediction",
    "Customer Segmentation",
    "Churn Analysis",
    "Recommendation Engine"
])

with tab1:
    st.header("Dashboard")
    st.metric("Total Customers", len(df))
    st.metric("Total Revenue", f"₹{df['Purchase_Amount'].sum():,.0f}")

with tab2:
    st.header("Prediction")
    st.write("Customer Segment Prediction")

with tab3:
    st.header("Customer Segmentation")

with tab4:
    st.header("Churn Analysis")

with tab5:
    st.header("Recommendation Engine")
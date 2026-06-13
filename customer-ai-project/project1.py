import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Customer Intelligence", layout="wide")

df = pd.read_csv("customer_dataset.csv")

st.title("📊 AI-Driven Customer Intelligence Platform")

# KPI Section
col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Average Budget", f"₹{df['monthly_spend'].mean():,.0f}")

with col3:
    st.metric("Total Revenue", f"₹{df['monthly_spend'].sum():,.0f}")

with col4:
    st.metric("Churn Rate", f"{df['churn'].mean()*100:.1f}%")

st.markdown("---")

# Charts
col1,col2 = st.columns(2)

with col1:
    st.subheader("Customer Segment Distribution")

    fig = px.pie(
        df,
        names="segment",
        title="Customer Segments"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Churn Distribution")

    fig = px.pie(
        df,
        names="churn",
        title="Churn Analysis"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

col1,col2 = st.columns(2)

with col1:
    st.subheader("Monthly Spend Distribution")

    fig = px.histogram(
        df,
        x="monthly_spend",
        nbins=20,
        title="Budget Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Age Distribution")

    fig = px.histogram(
        df,
        x="age",
        nbins=20,
        title="Age Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("Customer Dataset")

st.dataframe(df.head(20))
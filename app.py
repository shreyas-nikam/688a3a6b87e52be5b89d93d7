import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io # Although not directly used in app.py, included for consistency with page1.py

# Set a random seed for reproducibility
np.random.seed(42)

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

st.markdown("""
In this lab, we will explore the concept of model risk materiality in financial institutions using a multi-page Streamlit application. Users will be able to simulate various model scenarios to understand how different characteristics—such as complexity, data quality, usage frequency, and business impact—contribute to overall model risk and influence the required rigor of model risk management.

**Learning Goals:**
- Understand how model characteristics contribute to overall model risk.
- Learn to assess the materiality of a model based on its potential business impact.
- Explore how different levels of model risk necessitate varying degrees of risk management rigor.
- Understand the key insights contained in the provided document and supporting data through practical application.

This application is divided into three main pages:
- **Page 1: Data Generation & Validation**: Generate synthetic model data and perform initial data validation.
- **Page 2: Risk Score Calculation & Guidance**: Define model risk score calculation, assign management guidance based on materiality, and display results.
- **Page 3: Visualizations & Sensitivity Analysis**: Visualize model risk distributions and perform sensitivity analysis on key parameters.
""")

# Define factor mappings and score thresholds globally or in session state
if "factor_mappings" not in st.session_state:
    st.session_state.factor_mappings = {
        'complexity_level': {'Low': 1, 'Medium': 3, 'High': 5},
        'data_quality_index': lambda x: (100 - x) / 10,
        'usage_frequency': {'Low': 1, 'Medium': 3, 'High': 5},
        'business_impact_category': {'Low': 1, 'Medium': 3, 'High': 5, 'Critical': 10}
    }
if "score_thresholds" not in st.session_state:
    st.session_state.score_thresholds = {
        'Standard Oversight': 3,
        'Enhanced Scrutiny': 6,
        'Rigorous Management': np.inf
    }

page = st.sidebar.selectbox(label="Navigation", options=["Page 1: Data Generation & Validation", "Page 2: Risk Score & Guidance", "Page 3: Visualizations & Sensitivity Analysis"])

if page == "Page 1: Data Generation & Validation":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Page 2: Risk Score & Guidance":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Page 3: Visualizations & Sensitivity Analysis":
    from application_pages.page3 import run_page3
    run_page3()

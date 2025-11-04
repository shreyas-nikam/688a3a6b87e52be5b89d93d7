import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

def run_page1():
    st.markdown("""
    # Model Risk Materiality and Impact Simulator

    Welcome to this hands-on lab exploring the concept of model risk materiality in finance. In regulated financial institutions, models are integral to decision-making, from credit underwriting to risk management. However, models come with inherent risks, known as "model risk," defined as "the potential for adverse consequences from erroneous or misused model outputs" [1, Page 6].

    This notebook will guide you through simulating various model scenarios to understand how different model characteristics—such as complexity, data quality, usage frequency, and business impact—contribute to overall model risk and influence the required rigor of model risk management. A key concept we will explore is **materiality**, which dictates that "if models have less impact on a bank's financial condition, a less complex risk management approach may suffice. However, if models have a substantial business impact, the risk management framework should be more rigorous and extensive" [1, Page 8].

    By the end of this lab, you will:
    - Understand how model characteristics contribute to overall model risk.
    - Learn to assess the materiality of a model based on its potential business impact.
    - Explore how different levels of model risk necessitate varying degrees of risk management rigor.
    - Understand the key insights contained in the provided document and supporting data through practical application.
    """)

    st.markdown("## 2. Setting Up the Environment")
    st.markdown("""
    To begin, we need to import the necessary Python libraries. We will use `pandas` for data manipulation, `numpy` for numerical operations, `plotly.express` for plotting.
    """)

    st.markdown("The required libraries have been successfully imported, setting up our environment for data generation, analysis, and visualization.")

    st.markdown("## 3. Understanding Model Risk Factors")
    st.markdown("""
    Model risk is influenced by several intrinsic characteristics of a model and its operational context. As highlighted in the provided document, "Risk increases with complexity, uncertainty about inputs and assumptions, broader use, and larger potential impact" [1, Page 7]. We will quantify these factors using the following attributes for our synthetic models:

    1.  **Complexity Level**: The inherent intricacy of the model. More complex models often involve more assumptions and intricate logic, increasing potential for errors. (Categorical: Low, Medium, High)
    2.  **Data Quality Index**: A measure of the reliability, accuracy, and completeness of the input data. Poor data quality can lead to erroneous model outputs. (Numeric: 0-100, where higher is better quality)
    3.  **Usage Frequency**: How often the model is run or its outputs are used. Models with broader or more frequent use can have a wider impact if flawed. (Categorical: Low, Medium, High)
    4.  **Business Impact Category**: The potential financial or reputational consequences if the model's outputs are incorrect or misused. This is central to the concept of **materiality**. (Categorical: Low, Medium, High, Critical)

    These factors will be used to calculate a composite Model Risk Score, which will then inform management guidance based on materiality.
    """)

    st.markdown("## 4. Generating Synthetic Model Data")
    st.markdown("""
    We will generate a synthetic dataset representing various financial models, each with different attributes that influence its risk profile. This allows us to simulate different scenarios and observe their impact on model risk.

    The dataset will include the following columns:
    - `model_id`: A unique identifier for each synthetic model.
    - `complexity_level`: Categorical values ('Low', 'Medium', 'High').
    - `data_quality_index`: Numerical values ranging from 50 to 100, representing percentage quality.
    - `usage_frequency`: Categorical values ('Low', 'Medium', 'High').
    - `business_impact_category`: Categorical values ('Low', 'Medium', 'High', 'Critical').
    """)

    @st.cache_data  # Cache the dataframe generation for performance
    def generate_synthetic_model_data(num_models):
        """
        Generates a Pandas DataFrame of synthetic model attributes for risk analysis.
        """
        if not isinstance(num_models, int):
            raise TypeError("num_models must be an integer.")
        if num_models < 0:
            raise ValueError("num_models must be a non-negative integer.")

        expected_columns = [
            'model_id', 'complexity_level', 'data_quality_index',
            'usage_frequency', 'business_impact_category'
        ]

        if num_models == 0:
            return pd.DataFrame(columns=expected_columns)

        complexity_levels = ['Low', 'Medium', 'High']
        usage_frequencies = ['Low', 'Medium', 'High']
        business_impact_categories = ['Low', 'Medium', 'High', 'Critical']

        data = {
            'model_id': [f'model_{i:03d}' for i in np.arange(num_models)],
            'complexity_level': np.random.choice(complexity_levels, size=num_models),
            'data_quality_index': np.random.randint(50, 101, size=num_models),
            'usage_frequency': np.random.choice(usage_frequencies, size=num_models),
            'business_impact_category': np.random.choice(business_impact_categories, size=num_models)
        }

        df = pd.DataFrame(data, columns=expected_columns)
        df['complexity_level'] = pd.Categorical(df['complexity_level'], categories=complexity_levels, ordered=True)
        df['usage_frequency'] = pd.Categorical(df['usage_frequency'], categories=usage_frequencies, ordered=True)
        df['business_impact_category'] = pd.Categorical(df['business_impact_category'], categories=business_impact_categories, ordered=True)

        return df

    with st.sidebar:
        num_models_input = st.number_input("Number of Synthetic Models", min_value=0, max_value=1000, value=100, key="num_models_input")
        if st.button("Generate Synthetic Data", key="generate_data_button"):
            st.session_state.synthetic_models_df = generate_synthetic_model_data(num_models_input)
            st.success(f"Generated {num_models_input} synthetic models.")

    if 'synthetic_models_df' not in st.session_state or st.session_state.synthetic_models_df.empty:
        st.info("Please generate synthetic model data using the sidebar controls.")
    else:
        st.subheader("Head of the Synthetic Model Dataset:")
        st.dataframe(st.session_state.synthetic_models_df.head())
        st.subheader("DataFrame Info:")
        buffer = io.StringIO()
        st.session_state.synthetic_models_df.info(buf=buffer)
        st.text(buffer.getvalue())

    st.markdown("We have successfully generated a synthetic dataset containing entries, each representing a unique model with randomly assigned characteristics across complexity, data quality, usage frequency, and business impact. This dataset will serve as the basis for our model risk analysis.")

    st.markdown("## 5. Data Validation and Exploration")
    st.markdown("""
    Before proceeding with calculations, it's crucial to validate the dataset to ensure its integrity and consistency. This involves confirming expected column names and data types, checking for uniqueness of primary keys, and asserting no missing values in critical fields. We will also inspect summary statistics to understand the data's distribution.
    """)

    def validate_and_summarize_data(df):
        """
        Performs validation checks and prints summary statistics for the input DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            st.error(f"Error: Expected a Pandas DataFrame as input, but got {type(df).__name__}.")
            return

        expected_columns_list = [
            'model_id', 'complexity_level', 'data_quality_index',
            'usage_frequency', 'business_impact_category'
        ]
        expected_dtype_names = {
            'model_id': ['object'],
            'complexity_level': ['category'],
            'data_quality_index': ['int64'],
            'usage_frequency': ['category'],
            'business_impact_category': ['category']
        }
        critical_columns_for_na_check = expected_columns_list

        st.subheader("--- DataFrame Validation ---")

        df_columns = set(df.columns)
        expected_columns_set = set(expected_columns_list)

        missing_expected_columns = expected_columns_set - df_columns
        unexpected_columns = df_columns - expected_columns_set

        if missing_expected_columns:
            st.warning(f"Missing expected columns: {sorted(list(missing_expected_columns))}")
        else:
            st.success("All expected columns are present.")

        if unexpected_columns:
            st.warning(f"Unexpected columns found: {sorted(list(unexpected_columns))}")
        
        current_all_dtypes_match = True
        for col, expected_names in expected_dtype_names.items():
            if col in df.columns:
                actual_dtype_name = str(df[col].dtype)
                if not any(expected_name == actual_dtype_name for expected_name in expected_names):
                    st.warning(f"Mismatch in data type for column: {col} (expected {', '.join(expected_names)}, got {actual_dtype_name})")
                    current_all_dtypes_match = False
        
        if current_all_dtypes_match:
            st.success("All column data types are as expected.")

        if 'model_id' in df.columns:
            if df.empty:
                st.info("model_id is unique. (DataFrame is empty)")
            elif df['model_id'].duplicated().any():
                st.warning("Duplicate model_id values found.")
            else:
                st.success("model_id is unique.")

        found_missing_in_critical = []
        for col in critical_columns_for_na_check:
            if col in df.columns:
                if df[col].isnull().any():
                    found_missing_in_critical.append(col)
        
        if found_missing_in_critical:
            st.warning(f"Missing values found in critical columns: {sorted(found_missing_in_critical)}")
        else:
            st.success("No missing values found in critical columns.")

        st.subheader("\n--- Summary Statistics for DataFrame ---")

        if df.empty:
            st.info("DataFrame is empty.")
        
        numerical_df = df.select_dtypes(include=np.number)
        if not numerical_df.empty or (df.empty and not numerical_df.columns.empty):
            st.markdown("\nDescription for numerical columns:")
            st.dataframe(numerical_df.describe())
        elif numerical_df.empty and df.empty:
            st.markdown("\nNo numerical columns found for description.")
        
        categorical_df = df.select_dtypes(include='category')
        if not categorical_df.empty:
            st.markdown("\nValue counts for categorical columns:")
            for col in categorical_df.columns:
                st.markdown(f"\n--- {col} ---")
                st.dataframe(df[col].value_counts(dropna=False))
        elif categorical_df.empty and df.empty:
            st.markdown("\nNo categorical columns found.")

    if 'synthetic_models_df' in st.session_state and not st.session_state.synthetic_models_df.empty:
        validate_and_summarize_data(st.session_state.synthetic_models_df)
    elif 'synthetic_models_df' in st.session_state and st.session_state.synthetic_models_df.empty:
        st.info("No data available for validation. Please generate synthetic models.")

    st.markdown("The data validation checks confirm that our synthetic dataset has the expected structure and no critical missing values. The summary statistics provide initial insights into the distribution of model attributes, which aligns with our simulated data generation. This robust initial check ensures the reliability of subsequent analyses.")

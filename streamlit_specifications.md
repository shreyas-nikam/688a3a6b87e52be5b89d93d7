
# Streamlit Application Requirements Specification: Model Risk Materiality and Impact Simulator

## 1. Application Overview

This Streamlit application will provide an interactive platform for exploring the concept of model risk materiality in financial institutions. Users will simulate various model scenarios to understand how different characteristics—such as complexity, data quality, usage frequency, and business impact—contribute to overall model risk and influence the required rigor of model risk management.

**Learning Goals:**
- Understand how model characteristics contribute to overall model risk.
- Learn to assess the materiality of a model based on its potential business impact.
- Explore how different levels of model risk necessitate varying degrees of risk management rigor.
- Understand the key insights contained in the provided document and supporting data through practical application.

## 2. User Interface Requirements

### Layout and Navigation Structure
The application will feature a clear two-column layout:
- **Sidebar**: Will house all user input widgets and controls for model generation, risk factor adjustments, and sensitivity analysis.
- **Main Content Area**: Will display the introductory text, generated dataframes, summary statistics, and all interactive visualizations. Content will be presented sequentially, mirroring the Jupyter Notebook flow.

### Input Widgets and Controls
The sidebar will contain the following interactive elements:

1.  **Model Data Generation**:
    *   **Number of Synthetic Models**: `st.number_input` to specify `num_models` (default: 100, min: 0, max: 1000).
    *   **Generate Data Button**: `st.button` to trigger synthetic data generation and subsequent calculations.

2.  **Model Risk Score Configuration**:
    *   **Factor Weights**: `st.slider` or `st.number_input` widgets for each of the four weights ($w_C, w_{DQ}, w_{UF}, w_{BI}$), ensuring their sum is 1.0.
        *   `w_C` (Complexity Level Weight): Default 0.2
        *   `w_{DQ}` (Data Quality Index Weight): Default 0.2
        *   `w_{UF}` (Usage Frequency Weight): Default 0.1
        *   `w_{BI}` (Business Impact Category Weight): Default 0.5
    *   **Factor Mappings Display**: Read-only text or `st.markdown` to show the current mappings (e.g., complexity_level: {'Low': 1, 'Medium': 3, 'High': 5}).
    *   **Calculate Risk Scores Button**: `st.button` to re-calculate risk scores with updated weights.

3.  **Materiality Guidance Thresholds (Read-only)**:
    *   Display the defined `score_thresholds` for 'Standard Oversight', 'Enhanced Scrutiny', and 'Rigorous Management' using `st.markdown`.

4.  **Sensitivity Analysis Controls**:
    *   **Base Model Parameters**: A set of `st.selectbox` and `st.number_input` widgets to define a hypothetical base model for sensitivity analysis.
        *   `complexity_level`: `st.selectbox` ('Low', 'Medium', 'High') (default: 'Medium')
        *   `data_quality_index`: `st.number_input` (50-100) (default: 75)
        *   `usage_frequency`: `st.selectbox` ('Low', 'Medium', 'High') (default: 'Medium')
        *   `business_impact_category`: `st.selectbox` ('Low', 'Medium', 'High', 'Critical') (default: 'High')
    *   **Parameter to Vary**: `st.selectbox` to choose which factor to vary ('complexity_level', 'data_quality_index').
    *   **Variation Values**: Dynamic input based on `param_to_vary`.
        *   If 'complexity_level': `st.multiselect` ('Low', 'Medium', 'High')
        *   If 'data_quality_index': `st.slider` with range (50-100) and step 5, providing a list of values.
    *   **Run Sensitivity Analysis Button**: `st.button` to perform and display sensitivity analysis.

### Visualization Components
The main content area will display the following:

1.  **Dataframes**:
    *   **Synthetic Model Data**: `st.dataframe` to show `synthetic_models_df.head()` and `synthetic_models_df.info()`.
    *   **Models with Risk Scores and Guidance**: `st.dataframe` to show `models_with_guidance_df.head()` and `models_with_guidance_df['management_guidance'].value_counts()`.

2.  **Charts and Graphs**: All plots will be rendered using `st.pyplot`.
    *   **Aggregated Comparison Plot (Bar Chart)**: Average Model Risk Score by Business Impact Category.
    *   **Relationship Plot (Heatmap)**: Average Model Risk Score across Complexity Level and Business Impact Category.
    *   **Relationship Plot (Scatter Plot)**: Model Risk Score vs. Data Quality Index, colored by Complexity Level.
    *   **Sensitivity Analysis Line Plots**:
        *   Model Risk Score vs. Complexity Level.
        *   Model Risk Score vs. Data Quality Index.

### Interactive Elements and Feedback Mechanisms
-   Changes in input widgets will trigger automatic re-calculations and plot updates, or require explicit button presses (e.g., "Generate Data", "Calculate Risk Scores", "Run Sensitivity Analysis") for performance.
-   Tooltips or `st.info` messages will provide contextual information.
-   Error messages (e.g., if weights don't sum to 1, or if a column is missing) will be displayed using `st.error`.

## 3. Additional Requirements

### Annotation and Tooltip Specifications
-   **Risk Score Formula**: A tooltip or `st.markdown` section near the weights input explaining the formula: $$ \text{Model Risk Score} = w_C S_C + w_{DQ} S_{DQ} + w_{UF} S_{UF} + w_{BI} S_{BI} $$
-   **Factor Mappings**: Explanations for how each factor is scored (e.g., `data_quality_index` is inverted: `(100 - value) / 10`).
-   **Management Guidance Thresholds**: Clearly state the thresholds for 'Standard Oversight', 'Enhanced Scrutiny', and 'Rigorous Management'.
-   **Plot Descriptions**: Each visualization will have a clear title and accompanying descriptive text (from the Jupyter Notebook markdown) explaining what it shows and its significance.

### Save the states of the fields properly so that changes are not lost
-   All user inputs (number of models, factor weights, base model parameters, selected parameter to vary, variation values) must be stored in `st.session_state` to maintain their values across application reruns. This ensures a persistent user experience.
-   The generated `synthetic_models_df`, `models_with_risk_scores_df`, and `models_with_guidance_df` should also be stored in `st.session_state` to avoid regenerating data unnecessarily or losing calculated results upon widget interaction.

## 4. Notebook Content and Code Requirements

This section outlines the content and code stubs extracted from the Jupyter Notebook, detailing how they will be integrated into the Streamlit application.

### Initial Setup and Introduction
-   **Markdown Content (Introduction)**:
    ```python
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
    ```
-   **Markdown Content (Environment Setup)**:
    ```python
    st.markdown("## 2. Setting Up the Environment")
    st.markdown("""
    To begin, we need to import the necessary Python libraries. We will use `pandas` for data manipulation, `numpy` for numerical operations, `matplotlib.pyplot` for plotting, and `seaborn` for enhanced statistical visualizations.
    """)
    ```
-   **Code Stub (Library Imports and Plotting Configuration)**: These imports and configurations will be placed at the top of the Streamlit script.
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    # Configure plotting styles for better readability and color-blind friendliness
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

    # Set a random seed for reproducibility
    np.random.seed(42)
    ```
-   **Markdown Content (Setup Confirmation)**:
    ```python
    st.markdown("The required libraries have been successfully imported, setting up our environment for data generation, analysis, and visualization.")
    ```

### Understanding Model Risk Factors
-   **Markdown Content**:
    ```python
    st.markdown("## 3. Understanding Model Risk Factors")
    st.markdown("""
    Model risk is influenced by several intrinsic characteristics of a model and its operational context. As highlighted in the provided document, "Risk increases with complexity, uncertainty about inputs and assumptions, broader use, and larger potential impact" [1, Page 7]. We will quantify these factors using the following attributes for our synthetic models:

    1.  **Complexity Level**: The inherent intricacy of the model. More complex models often involve more assumptions and intricate logic, increasing potential for errors. (Categorical: Low, Medium, High)
    2.  **Data Quality Index**: A measure of the reliability, accuracy, and completeness of the input data. Poor data quality can lead to erroneous model outputs. (Numeric: 0-100, where higher is better quality)
    3.  **Usage Frequency**: How often the model is run or its outputs are used. Models with broader or more frequent use can have a wider impact if flawed. (Categorical: Low, Medium, High)
    4.  **Business Impact Category**: The potential financial or reputational consequences if the model's outputs are incorrect or misused. This is central to the concept of **materiality**. (Categorical: Low, Medium, High, Critical)

    These factors will be used to calculate a composite Model Risk Score, which will then inform management guidance based on materiality.
    """)
    ```

### Generating Synthetic Model Data
-   **Markdown Content**:
    ```python
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
    ```
-   **Code Stub (`generate_synthetic_model_data` function)**: This function will be called when the user triggers data generation.
    ```python
    @st.cache_data # Cache the dataframe generation for performance
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
    ```
-   **Code Stub (Calling `generate_synthetic_model_data` and displaying head/info)**:
    ```python
    # Example call within Streamlit's main logic
    # num_models_input = st.sidebar.number_input("Number of Synthetic Models", min_value=0, max_value=1000, value=100)
    # if st.sidebar.button("Generate Synthetic Data"):
    #     st.session_state.synthetic_models_df = generate_synthetic_model_data(num_models_input)
    #
    # if 'synthetic_models_df' in st.session_state and not st.session_state.synthetic_models_df.empty:
    #     st.subheader("Head of the Synthetic Model Dataset:")
    #     st.dataframe(st.session_state.synthetic_models_df.head())
    #     st.subheader("DataFrame Info:")
    #     # Use a string buffer to capture info() output
    #     import io
    #     buffer = io.StringIO()
    #     st.session_state.synthetic_models_df.info(buf=buffer)
    #     st.text(buffer.getvalue())
    ```
-   **Markdown Content (Generation Confirmation)**:
    ```python
    st.markdown("We have successfully generated a synthetic dataset containing 100 entries, each representing a unique model with randomly assigned characteristics across complexity, data quality, usage frequency, and business impact. This dataset will serve as the basis for our model risk analysis.")
    ```

### Data Validation and Exploration
-   **Markdown Content**:
    ```python
    st.markdown("## 5. Data Validation and Exploration")
    st.markdown("""
    Before proceeding with calculations, it's crucial to validate the dataset to ensure its integrity and consistency. This involves confirming expected column names and data types, checking for uniqueness of primary keys, and asserting no missing values in critical fields. We will also inspect summary statistics to understand the data's distribution.
    """)
    ```
-   **Code Stub (`validate_and_summarize_data` function)**:
    ```python
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
    ```
-   **Code Stub (Calling `validate_and_summarize_data`)**:
    ```python
    # if 'synthetic_models_df' in st.session_state:
    #     validate_and_summarize_data(st.session_state.synthetic_models_df)
    ```
-   **Markdown Content (Validation Confirmation)**:
    ```python
    st.markdown("The data validation checks confirm that our synthetic dataset has the expected structure and no critical missing values. The summary statistics provide initial insights into the distribution of model attributes, which aligns with our simulated data generation. This robust initial check ensures the reliability of subsequent analyses.")
    ```

### Defining the Model Risk Score Calculation
-   **Markdown Content**:
    ```python
    st.markdown("## 6. Defining the Model Risk Score Calculation")
    st.markdown("""
    To quantify model risk, we will implement a simplified weighted sum methodology. Each model characteristic (complexity, data quality, usage frequency, business impact) will be assigned a numerical score, and these scores will be combined using predefined weights to calculate a composite `model_risk_score`. This score reflects the overall risk profile of a model, aligning with the principle that "risk increases with complexity, uncertainty about inputs and assumptions, broader use, and larger potential impact" [1, Page 7].

    The formula for the Model Risk Score is:
    $$ \\text{Model Risk Score} = w_C S_C + w_{DQ} S_{DQ} + w_{UF} S_{UF} + w_{BI} S_{BI} $$
    Where:
    - $S_C$: Score for Complexity Level
    - $S_{DQ}$: Score for Data Quality Index (inverted, higher quality = lower risk score)
    - $S_{UF}$: Score for Usage Frequency
    - $S_{BI}$: Score for Business Impact Category
    - $w_C, w_{DQ}, w_{UF}, w_{BI}$: Respective weights for each factor, summing to 1.

    We will use the following mappings and weights:

    **Factor Mappings:**
    - `complexity_level`: {'Low': 1, 'Medium': 3, 'High': 5}
    - `data_quality_index`: `(100 - value) / 10` (transforms 0-100 quality to 0-10 risk score)
    - `usage_frequency`: {'Low': 1, 'Medium': 3, 'High': 5}
    - `business_impact_category`: {'Low': 1, 'Medium': 3, 'High': 5, 'Critical': 10}

    **Weights:**
    - Complexity: $w_C = 0.2$
    - Data Quality: $w_{DQ} = 0.2$
    - Usage Frequency: $w_{UF} = 0.1$
    - Business Impact: $w_{BI} = 0.5$
    """)
    ```
-   **Code Stub (Weights and Factor Mappings Definition)**: These will be defined globally or in `st.session_state`.
    ```python
    # Define the weights for each model risk factor (configurable via sidebar)
    # weights = {
    #     'complexity_level': st.session_state.wc_weight,
    #     'data_quality_index': st.session_state.wdq_weight,
    #     'usage_frequency': st.session_state.wuf_weight,
    #     'business_impact_category': st.session_state.wbi_weight
    # }

    # Define the factor mappings (static for this app)
    factor_mappings = {
        'complexity_level': {'Low': 1, 'Medium': 3, 'High': 5},
        'data_quality_index': lambda x: (100 - x) / 10,
        'usage_frequency': {'Low': 1, 'Medium': 3, 'High': 5},
        'business_impact_category': {'Low': 1, 'Medium': 3, 'High': 5, 'Critical': 10}
    }
    ```
-   **Code Stub (`calculate_model_risk_score` function)**:
    ```python
    def calculate_model_risk_score(df, weights, factor_mappings):
        """Computes a model risk score for each model in the DataFrame."""
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError(f"Weights must sum to 1.0. Current sum: {sum(weights.values())}")

        df_output = df.copy()
        df_output['model_risk_score'] = 0.0

        for factor, weight in weights.items():
            if factor not in df_output.columns:
                raise KeyError(f"Missing required column: '{factor}' in the input DataFrame.")

            mapping = factor_mappings.get(factor)
            if mapping is None:
                raise ValueError(f"No mapping defined in 'factor_mappings' for factor: '{factor}'.")

            factor_scores = None
            if isinstance(mapping, dict):
                try:
                    factor_scores = df_output[factor].apply(lambda x: mapping[x])
                except KeyError as e:
                    raise KeyError(f"Unknown categorical value '{e.args[0]}' encountered for factor '{factor}'. "
                                   f"Expected one of: {list(mapping.keys())}.") from e
            elif callable(mapping):
                factor_scores = df_output[factor].apply(mapping)
            else:
                raise TypeError(f"Unsupported mapping type for factor '{factor}'. "
                                "Mapping must be a dictionary for categorical values or a callable for continuous values.")

            df_output['model_risk_score'] += factor_scores * weight

        return df_output
    ```
-   **Code Stub (Calling `calculate_model_risk_score` and displaying head)**:
    ```python
    # Example call in main Streamlit logic:
    # if 'synthetic_models_df' in st.session_state:
    #     current_weights = {
    #         'complexity_level': st.session_state.wc_weight,
    #         'data_quality_index': st.session_state.wdq_weight,
    #         'usage_frequency': st.session_state.wuf_weight,
    #         'business_impact_category': st.session_state.wbi_weight
    #     }
    #     st.session_state.models_with_risk_scores_df = calculate_model_risk_score(st.session_state.synthetic_models_df, current_weights, factor_mappings)
    #     st.subheader("Head of the DataFrame with model risk scores:")
    #     st.dataframe(st.session_state.models_with_risk_scores_df.head())
    ```
-   **Markdown Content (Calculation Confirmation)**:
    ```python
    st.markdown("The `model_risk_score` has been successfully calculated for each synthetic model based on the defined weighted sum methodology. This score now provides a quantitative measure of the risk associated with each model, considering its inherent characteristics and operational context.")
    ```

### Assessing Materiality and Management Guidance
-   **Markdown Content**:
    ```python
    st.markdown("## 7. Assessing Materiality and Management Guidance")
    st.markdown("""
    The concept of **materiality** is critical in model risk management, guiding the intensity of oversight required. As described in the document, "Materiality is crucial in model risk management. If models have less impact on a bank's financial condition, a less complex risk management approach may suffice. However, if models have a substantial business impact, the risk management framework should be more rigorous and extensive" [1, Page 8].

    Based on the calculated `model_risk_score`, we will classify models into different management guidance levels. These levels correspond to increasing rigor of risk management practices.

    **Risk Score Thresholds and Guidance:**
    - `model_risk_score` $ \\le 3$: 'Standard Oversight' (Less complex management approach)
    - $3 < \\text{model\\_risk\\_score} \\le 6$: 'Enhanced Scrutiny' (Requires more attention and review)
    - `model_risk_score` $ > 6$: 'Rigorous Management' (More rigorous and extensive framework)
    """)
    ```
-   **Code Stub (Score Thresholds Definition)**: Will be defined globally or in `st.session_state`.
    ```python
    # Define the score thresholds for different management guidance levels
    score_thresholds = {
        'Standard Oversight': 3,
        'Enhanced Scrutiny': 6,
        'Rigorous Management': np.inf
    }
    ```
-   **Code Stub (`provide_materiality_guidance` function)**:
    ```python
    def provide_materiality_guidance(df, score_thresholds):
        """
        Assigns a risk management guidance level to each model based on its calculated model risk score and predefined thresholds.
        """
        if df.empty:
            df['management_guidance'] = pd.Series([], dtype=str)
            return df

        conditions = [
            df['model_risk_score'] <= score_thresholds['Standard Oversight'],
            (df['model_risk_score'] > score_thresholds['Standard Oversight']) & (df['model_risk_score'] <= score_thresholds['Enhanced Scrutiny']),
            df['model_risk_score'] > score_thresholds['Enhanced Scrutiny']
        ]

        choices = [
            'Standard Oversight',
            'Enhanced Scrutiny',
            'Rigorous Management'
        ]

        df['management_guidance'] = np.select(conditions, choices)
        df['management_guidance'] = df['management_guidance'].astype(str)

        return df
    ```
-   **Code Stub (Calling `provide_materiality_guidance` and displaying head/value_counts)**:
    ```python
    # Example call in main Streamlit logic:
    # if 'models_with_risk_scores_df' in st.session_state:
    #     st.session_state.models_with_guidance_df = provide_materiality_guidance(st.session_state.models_with_risk_scores_df, score_thresholds)
    #     st.subheader("Head of the DataFrame with management guidance:")
    #     st.dataframe(st.session_state.models_with_guidance_df.head())
    #     st.subheader("Distribution of Management Guidance Levels:")
    #     st.dataframe(st.session_state.models_with_guidance_df['management_guidance'].value_counts())
    ```
-   **Markdown Content (Guidance Confirmation)**:
    ```python
    st.markdown("Each model has now been assigned a `management_guidance` level, which directly reflects its materiality based on the calculated model risk score. This provides actionable insight into the intensity of risk management required, demonstrating how quantitative risk assessment translates into practical oversight strategies.")
    ```

### Visualizing Model Risk Distribution by Business Impact
-   **Markdown Content**:
    ```python
    st.markdown("## 8. Visualizing Model Risk Distribution by Business Impact")
    st.markdown("""
    Understanding how model risk varies across different `business_impact_category` levels is crucial for prioritizing risk management efforts. A bar plot will allow us to visualize the average model risk score for each business impact category, clearly showing which categories pose higher inherent risks. This directly ties back to materiality, where higher business impact generally implies higher risk and necessitates more rigorous management.
    """)
    ```
-   **Code Stub (`plot_risk_by_category` function)**:
    ```python
    def plot_risk_by_category(df, category_col, score_col, plot_type='bar'):
        """
        Generates an aggregated comparison plot (e.g., bar chart) of model risk scores across
        different categories.
        """
        if category_col not in df.columns:
            raise KeyError(f"Column '{category_col}' not found in the DataFrame.")
        if score_col not in df.columns:
            raise KeyError(f"Column '{score_col}' not found in the DataFrame.")

        valid_plot_types = ['bar']
        if plot_type not in valid_plot_types:
            raise ValueError(f"plot_type must be '{' or '.join(valid_plot_types)}'")

        aggregated_df = df.groupby(category_col, observed=False)[score_col].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == 'bar':
            sns.barplot(x=category_col, y=score_col, data=aggregated_df, ax=ax, palette="viridis")
            ax.set_title(f'Average Model Risk Score by {category_col.replace("_", " ").title()}')
            ax.set_xlabel(category_col.replace("_", " ").title())
            ax.set_ylabel(f'Average {score_col.replace("_", " ").title()}')
            if not aggregated_df.empty:
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig # Return figure for st.pyplot
    ```
-   **Code Stub (Calling `plot_risk_by_category`)**:
    ```python
    # if 'models_with_guidance_df' in st.session_state:
    #     st.subheader("Average Model Risk Score by Business Impact Category")
    #     fig_bar = plot_risk_by_category(st.session_state.models_with_guidance_df, 'business_impact_category', 'model_risk_score', plot_type='bar')
    #     st.pyplot(fig_bar)
    ```
-   **Markdown Content (Bar Plot Insight)**:
    ```python
    st.markdown("The bar plot clearly illustrates a direct correlation between the `business_impact_category` and the `average model_risk_score`. Models categorized with 'Critical' business impact exhibit significantly higher average risk scores, underscoring the importance of materiality in focusing risk management resources where potential adverse consequences are greatest.")
    ```

### Visualizing Model Risk across Complexity and Business Impact (Heatmap)
-   **Markdown Content**:
    ```python
    st.markdown("## 9. Visualizing Model Risk across Complexity and Business Impact (Heatmap)")
    st.markdown("""
    While individual factors contribute to risk, understanding their combined effect is vital. A heatmap provides an excellent way to visualize the average `model_risk_score` across two categorical dimensions simultaneously: `complexity_level` and `business_impact_category`. This allows us to identify specific combinations of model characteristics that result in the highest risk concentrations.
    """)
    ```
-   **Code Stub (`plot_risk_heatmap` function)**:
    ```python
    def plot_risk_heatmap(df, x_col, y_col, value_col):
        """
        Generates a heatmap showing the average model risk score across two categorical dimensions.
        """
        for col in [x_col, y_col, value_col]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in the DataFrame.")

        try:
            aggregated_data = df.groupby([y_col, x_col], observed=False)[value_col].mean().reset_index()
        except TypeError as e:
            raise TypeError(f"Cannot calculate mean on non-numeric column '{value_col}'. "
                            "Ensure 'value_col' contains numeric data. Error: {e}") from e

        pivot_table = aggregated_data.pivot(index=y_col, columns=x_col, values=value_col)

        fig, ax = plt.subplots(figsize=(10, 7)) 
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, ax=ax)

        ax.set_title(
            f'Average {value_col.replace("_", " ").title()} by '
            f'{y_col.replace("_", " ").title()} and {x_col.replace("_", " ").title()}',
            fontsize=14
        )
        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)

        plt.tight_layout()
        return fig
    ```
-   **Code Stub (Calling `plot_risk_heatmap`)**:
    ```python
    # if 'models_with_guidance_df' in st.session_state:
    #     st.subheader("Average Model Risk Score Heatmap")
    #     fig_heatmap = plot_risk_heatmap(st.session_state.models_with_guidance_df, 'complexity_level', 'business_impact_category', 'model_risk_score')
    #     st.pyplot(fig_heatmap)
    ```
-   **Markdown Content (Heatmap Insight)**:
    ```python
    st.markdown("The heatmap reveals distinct patterns, showing that models with 'High' complexity and 'Critical' business impact consistently have the highest average `model_risk_score`. This visualization effectively highlights that model risk is not solely an additive function but can be exacerbated by the interaction of multiple high-risk attributes, guiding more targeted risk mitigation strategies.")
    ```

### Relationship between Input Parameters and Model Risk (Scatter Plot)
-   **Markdown Content**:
    ```python
    st.markdown("## 10. Relationship between Input Parameters and Model Risk (Scatter Plot)")
    st.markdown("""
    A scatter plot can help us explore the relationship between a continuous input parameter and the calculated `model_risk_score`. Here, we will visualize how `data_quality_index` influences `model_risk_score`, with points colored by `complexity_level`. We expect to see lower data quality (lower index) generally leading to higher risk scores, and potentially observing how complexity affects this relationship.
    """)
    ```
-   **Code Stub (`plot_risk_relationship` function)**:
    ```python
    def plot_risk_relationship(df, x_col, y_col, hue_col=None):
        """
        Generates a scatter plot to visualize the relationship between two parameters,
        optionally colored by a third categorical parameter.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a Pandas DataFrame.")

        for col in [x_col, y_col]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        if hue_col is not None and hue_col not in df.columns:
            raise KeyError(f"Column '{hue_col}' not found in DataFrame.")

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, palette="viridis", s=100)

        title = f'Relationship between {x_col.replace("_", " ").title()} and {y_col.replace("_", " ").title()}'
        if hue_col:
            title += f' by {hue_col.replace("_", " ").title()}'
        ax.set_title(title)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())

        if hue_col:
            ax.legend(title=hue_col.replace("_", " ").title(), bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    ```
-   **Code Stub (Calling `plot_risk_relationship`)**:
    ```python
    # if 'models_with_guidance_df' in st.session_state:
    #     st.subheader("Model Risk Score vs. Data Quality Index")
    #     fig_scatter = plot_risk_relationship(st.session_state.models_with_guidance_df, 'data_quality_index', 'model_risk_score', hue_col='complexity_level')
    #     st.pyplot(fig_scatter)
    ```
-   **Markdown Content (Scatter Plot Insight)**:
    ```python
    st.markdown("The scatter plot demonstrates the expected inverse relationship: as `data_quality_index` decreases (indicating poorer data quality), the `model_risk_score` tends to increase. Furthermore, models with 'High' complexity often exhibit higher risk scores across similar data quality levels compared to 'Low' or 'Medium' complexity models, reinforcing the multi-factorial nature of model risk.")
    ```

### Sensitivity Analysis: Impact of Complexity on Model Risk
-   **Markdown Content**:
    ```python
    st.markdown("## 11. Sensitivity Analysis: Impact of Complexity on Model Risk")
    st.markdown("""
    Sensitivity analysis allows us to understand how changes in a single input parameter affect the `model_risk_score`, while holding other factors constant. This provides insights into the relative importance and leverage points for risk mitigation.

    We will simulate a hypothetical model and vary its `complexity_level` while keeping its `data_quality_index`, `usage_frequency`, and `business_impact_category` constant. This will show us how the `model_risk_score` responds to changes in complexity.
    """)
    ```
-   **Code Stub (`perform_sensitivity_analysis` function)**:
    ```python
    def perform_sensitivity_analysis(base_model_params, param_to_vary, variation_values, weights, factor_mappings):
        """
        Conducts sensitivity analysis by varying a single model parameter while keeping others constant,
        calculating the resulting model risk scores.
        """
        if param_to_vary not in weights:
            raise KeyError(f"Parameter to vary '{param_to_vary}' is not defined in the 'weights' dictionary, "
                           f"and therefore cannot be analyzed as a factor contributing to the model risk score.")
        if param_to_vary not in factor_mappings:
            raise KeyError(f"Parameter to vary '{param_to_vary}' is not defined in the 'factor_mappings' dictionary, "
                           f"and therefore cannot be analyzed as a factor contributing to the model risk score.")

        results_data = []

        if not variation_values:
            return pd.DataFrame(columns=[param_to_vary, 'model_risk_score'])

        for value in variation_values:
            current_params = base_model_params.copy()
            current_params[param_to_vary] = value

            model_risk_score = 0.0

            for factor, weight in weights.items():
                factor_value = current_params[factor]
                mapper = factor_mappings[factor]

                mapped_value = None
                if isinstance(mapper, dict):
                    mapped_value = mapper[factor_value]
                elif callable(mapper):
                    mapped_value = mapper(factor_value)
                else:
                    raise TypeError(f"Unsupported mapping type for factor '{factor}': {type(mapper)}. "
                                    f"Expected dict or callable.")
                
                model_risk_score += weight * mapped_value
            
            results_data.append({param_to_vary: value, 'model_risk_score': model_risk_score})

        return pd.DataFrame(results_data)
    ```
-   **Code Stub (Calling `perform_sensitivity_analysis` for complexity and plotting)**:
    ```python
    # # Example call in main Streamlit logic:
    # # base_model_params, current_weights (from session state), factor_mappings (global)
    #
    # st.subheader("Sensitivity Analysis: Complexity Level")
    # complexity_variation_values = ['Low', 'Medium', 'High'] # from st.multiselect
    # complexity_sensitivity_df = perform_sensitivity_analysis(
    #     base_model_params_input, # from st.selectbox/number_input
    #     'complexity_level',
    #     complexity_variation_values,
    #     current_weights,
    #     factor_mappings
    # )
    #
    # fig_comp_sens, ax = plt.subplots(figsize=(10, 6))
    # sns.lineplot(x='complexity_level', y='model_risk_score', data=complexity_sensitivity_df, marker='o', ax=ax)
    # ax.set_title('Sensitivity of Model Risk Score to Complexity Level')
    # ax.set_xlabel('Complexity Level')
    # ax.set_ylabel('Model Risk Score')
    # plt.grid(True)
    # plt.tight_layout()
    # st.pyplot(fig_comp_sens)
    ```
-   **Markdown Content (Complexity Sensitivity Insight)**:
    ```python
    st.markdown("The sensitivity analysis clearly shows that increasing a model's `complexity_level` (from Low to High) directly leads to a higher `model_risk_score`, assuming all other factors remain constant. This highlights that model complexity is a significant driver of risk, and managing complexity can be an effective strategy for reducing overall model risk.")
    ```

### Sensitivity Analysis: Impact of Data Quality on Model Risk
-   **Markdown Content**:
    ```python
    st.markdown("## 12. Sensitivity Analysis: Impact of Data Quality on Model Risk")
    st.markdown("""
    Continuing our sensitivity analysis, we now examine the impact of `data_quality_index` on the `model_risk_score`. We will vary the `data_quality_index` for our hypothetical model, keeping `complexity_level`, `usage_frequency`, and `business_impact_category` fixed. This analysis will underscore the importance of robust data governance and quality assurance practices in managing model risk.
    """)
    ```
-   **Code Stub (Calling `perform_sensitivity_analysis` for data quality and plotting)**:
    ```python
    # # Example call in main Streamlit logic:
    # # data_quality_variation_values = list(range(st.session_state.min_dq, st.session_state.max_dq + 1, st.session_state.step_dq))
    #
    # st.subheader("Sensitivity Analysis: Data Quality Index")
    # data_quality_sensitivity_df = perform_sensitivity_analysis(
    #     base_model_params_input,
    #     'data_quality_index',
    #     data_quality_variation_values,
    #     current_weights,
    #     factor_mappings
    # )
    #
    # fig_dq_sens, ax = plt.subplots(figsize=(10, 6))
    # sns.lineplot(x='data_quality_index', y='model_risk_score', data=data_quality_sensitivity_df, marker='o', ax=ax)
    # ax.set_title('Sensitivity of Model Risk Score to Data Quality Index')
    # ax.set_xlabel('Data Quality Index (Higher is Better)')
    # ax.set_ylabel('Model Risk Score')
    # plt.grid(True)
    # plt.tight_layout()
    # st.pyplot(fig_dq_sens)
    ```
-   **Markdown Content (Data Quality Sensitivity Insight)**:
    ```python
    st.markdown("The sensitivity analysis demonstrates that improving `data_quality_index` (increasing the index value) leads to a reduction in the `model_risk_score`. Conversely, poorer data quality significantly elevates the risk. This underscores that investment in data quality initiatives is a direct and impactful way to mitigate model risk.")
    ```

### Conclusion and Key Takeaways
-   **Markdown Content**:
    ```python
    st.markdown("## 13. Conclusion and Key Takeaways")
    st.markdown("""
    This hands-on lab has provided a practical simulation of model risk assessment, emphasizing the critical role of **materiality** in financial institutions. We started by generating synthetic model data and then quantified model risk based on key characteristics like complexity, data quality, usage frequency, and business impact.

    Key takeaways from this exercise include:
    - **Multi-factorial Nature of Risk**: Model risk is a composite of various factors, and changes in any one of them can significantly alter a model's overall risk profile.
    - **Materiality Drives Management**: The concept of materiality, often driven by potential business impact, directly dictates the rigor and intensity of model risk management practices required, aligning with regulatory expectations.
    - **Data-Driven Insights**: Visualizations and sensitivity analyses provide powerful tools to understand the relationships between model attributes and their risk implications, allowing for targeted risk mitigation strategies.
    - **Actionable Guidance**: By calculating a model risk score and translating it into practical management guidance, we demonstrated a tangible framework for model risk governance, as outlined in the provided document [1].

    This simulation reinforces the understanding that effective model risk management is dynamic, requiring continuous assessment and adaptation based on a model's evolving characteristics and its potential impact on the institution.
    """)
    ```

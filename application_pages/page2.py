import streamlit as st
import pandas as pd
import numpy as np

def calculate_model_risk_score(df, weights, factor_mappings):
    """Computes a model risk score for each model in the DataFrame."""
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"Weights must sum to 1.0. Current sum: {sum(weights.values())}")

    df_output = df.copy()
    df_output['model_risk_score'] = 0.0

    for factor, weight in weights.items():
        if factor not in df_output.columns:
            raise KeyError(f"Missing required column: \'{factor}\' in the input DataFrame.")

        mapping = factor_mappings.get(factor)
        if mapping is None:
            raise ValueError(f"No mapping defined in \'factor_mappings\' for factor: \'{factor}\'.")

        factor_scores = None
        if isinstance(mapping, dict):
            try:
                factor_scores = df_output[factor].apply(lambda x: mapping[x])
            except KeyError as e:
                raise KeyError(f"Unknown categorical value \'{e.args[0]}\' encountered for factor \'{factor}\'. "
                               f"Expected one of: {list(mapping.keys())}.") from e
        elif callable(mapping):
            factor_scores = df_output[factor].apply(mapping)
        else:
            raise TypeError(f"Unsupported mapping type for factor \'{factor}\'. "
                            "Mapping must be a dictionary for categorical values or a callable for continuous values.")

        df_output['model_risk_score'] += factor_scores * weight

    return df_output

def provide_materiality_guidance(df, score_thresholds):
    """
    Assigns a risk management guidance level to each model based on its calculated model risk score and predefined thresholds.
    """
    df_copy = df.copy() # Operate on a copy to avoid modifying original dataframe

    if df_copy.empty or 'model_risk_score' not in df_copy.columns:
        df_copy['management_guidance'] = pd.Series([], dtype=str)
        return df_copy

    conditions = [
        df_copy['model_risk_score'] <= score_thresholds['Standard Oversight'],
        (df_copy['model_risk_score'] > score_thresholds['Standard Oversight']) & (df_copy['model_risk_score'] <= score_thresholds['Enhanced Scrutiny']),
        df_copy['model_risk_score'] > score_thresholds['Enhanced Scrutiny']
    ]

    choices = [
        'Standard Oversight',
        'Enhanced Scrutiny',
        'Rigorous Management'
    ]

    df_copy['management_guidance'] = np.select(conditions, choices)
    df_copy['management_guidance'] = df_copy['management_guidance'].astype(str)

    return df_copy

def run_page2():
    st.markdown("## 6. Defining the Model Risk Score Calculation")
    st.markdown("""
    To quantify model risk, we will implement a simplified weighted sum methodology. Each model characteristic (complexity, data quality, usage frequency, business impact) will be assigned a numerical score, and these scores will be combined using predefined weights to calculate a composite `model_risk_score`. This score reflects the overall risk profile of a model, aligning with the principle that "risk increases with complexity, uncertainty about inputs and assumptions, broader use, and larger potential impact" [1, Page 7].

    The formula for the Model Risk Score is:
    $$ \text{Model Risk Score} = w_C S_C + w_{DQ} S_{DQ} + w_{UF} S_{UF} + w_{BI} S_{BI} $$
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

    **Default Weights:**
    - Complexity: $w_C = 0.2$
    - Data Quality: $w_{DQ} = 0.2$
    - Usage Frequency: $w_{UF} = 0.1$
    - Business Impact: $w_{BI} = 0.5$
    """)

    if 'synthetic_models_df' not in st.session_state or st.session_state.synthetic_models_df.empty:
        st.warning("Please go to 'Page 1: Data Generation & Validation' to generate synthetic model data first.")
        return

    st.sidebar.subheader("Model Risk Score Configuration")
    st.sidebar.markdown(r"**Factor Weights (Sum must be 1.0):**")

    # Initialize weights in session_state if not present
    if "wc_weight" not in st.session_state:
        st.session_state.wc_weight = 0.2
    if "wdq_weight" not in st.session_state:
        st.session_state.wdq_weight = 0.2
    if "wuf_weight" not in st.session_state:
        st.session_state.wuf_weight = 0.1
    if "wbi_weight" not in st.session_state:
        st.session_state.wbi_weight = 0.5

    wc = st.sidebar.slider("Complexity Level Weight ($w_C$)", 0.0, 1.0, st.session_state.wc_weight, 0.05, key="wc_slider")
    wdq = st.sidebar.slider("Data Quality Index Weight ($w_{DQ}$)", 0.0, 1.0, st.session_state.wdq_weight, 0.05, key="wdq_slider")
    wuf = st.sidebar.slider("Usage Frequency Weight ($w_{UF}$)", 0.0, 1.0, st.session_state.wuf_weight, 0.05, key="wuf_slider")
    wbi = st.sidebar.slider("Business Impact Category Weight ($w_{BI}$)", 0.0, 1.0, st.session_state.wbi_weight, 0.05, key="wbi_slider")

    current_weights_sum = wc + wdq + wuf + wbi
    st.sidebar.markdown(f"**Current Weights Sum:** {current_weights_sum:.2f}")

    if not np.isclose(current_weights_sum, 1.0):
        st.sidebar.error("Weights must sum to 1.0!")
    else:
        st.session_state.wc_weight = wc
        st.session_state.wdq_weight = wdq
        st.session_state.wuf_weight = wuf
        st.session_state.wbi_weight = wbi
        if st.sidebar.button("Calculate Risk Scores", key="calculate_risk_button"):
            current_weights = {
                'complexity_level': st.session_state.wc_weight,
                'data_quality_index': st.session_state.wdq_weight,
                'usage_frequency': st.session_state.wuf_weight,
                'business_impact_category': st.session_state.wbi_weight
            }
            st.session_state.models_with_risk_scores_df = calculate_model_risk_score(st.session_state.synthetic_models_df, current_weights, st.session_state.factor_mappings)
            st.success("Model risk scores calculated!")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Factor Mappings Display:**")
    st.sidebar.markdown(r"- `complexity_level`: {'Low': 1, 'Medium': 3, 'High': 5}")
    st.sidebar.markdown(r"- `data_quality_index`: `(100 - value) / 10`")
    st.sidebar.markdown(r"- `usage_frequency`: {'Low': 1, 'Medium': 3, 'High': 5}")
    st.sidebar.markdown(r"- `business_impact_category`: {'Low': 1, 'Medium': 3, 'High': 5, 'Critical': 10}")


    if 'models_with_risk_scores_df' in st.session_state and not st.session_state.models_with_risk_scores_df.empty:
        st.subheader("Head of the DataFrame with model risk scores:")
        st.dataframe(st.session_state.models_with_risk_scores_df.head())
    else:
        st.info("No model risk scores calculated yet. Adjust weights in the sidebar and click 'Calculate Risk Scores'.")

    st.markdown("The `model_risk_score` has been successfully calculated for each synthetic model based on the defined weighted sum methodology. This score now provides a quantitative measure of the risk associated with each model, considering its inherent characteristics and operational context.")

    st.markdown("## 7. Assessing Materiality and Management Guidance")
    st.markdown("""
    The concept of **materiality** is critical in model risk management, guiding the intensity of oversight required. As described in the document, "Materiality is crucial in model risk management. If models have less impact on a bank's financial condition, a less complex risk management approach may suffice. However, if models have a substantial business impact, the risk management framework should be more rigorous and extensive" [1, Page 8].

    Based on the calculated `model_risk_score`, we will classify models into different management guidance levels. These levels correspond to increasing rigor of risk management practices.

    **Risk Score Thresholds and Guidance:**
    - `model_risk_score` $ \le 3$: 'Standard Oversight' (Less complex management approach)
    - $3 < \text{model\_risk\_score} \le 6$: 'Enhanced Scrutiny' (Requires more attention and review)
    - `model_risk_score` $ > 6$: 'Rigorous Management' (More rigorous and extensive framework)
    """)
    
    st.sidebar.subheader("Materiality Guidance Thresholds (Read-only)")
    st.sidebar.markdown(f"- Standard Oversight: $\le$ {st.session_state.score_thresholds['Standard Oversight']}")
    st.sidebar.markdown(f"- Enhanced Scrutiny: > {st.session_state.score_thresholds['Standard Oversight']} and $\le$ {st.session_state.score_thresholds['Enhanced Scrutiny']}")
    st.sidebar.markdown(f"- Rigorous Management: > {st.session_state.score_thresholds['Enhanced Scrutiny']}")

    if 'models_with_risk_scores_df' in st.session_state and not st.session_state.models_with_risk_scores_df.empty:
        st.session_state.models_with_guidance_df = provide_materiality_guidance(st.session_state.models_with_risk_scores_df, st.session_state.score_thresholds)
        st.subheader("Head of the DataFrame with management guidance:")
        st.dataframe(st.session_state.models_with_guidance_df.head())
        st.subheader("Distribution of Management Guidance Levels:")
        st.dataframe(st.session_state.models_with_guidance_df['management_guidance'].value_counts())
    else:
        st.info("No management guidance generated yet. Calculate risk scores first.")

    st.markdown("Each model has now been assigned a `management_guidance` level, which directly reflects its materiality based on the calculated model risk score. This provides actionable insight into the intensity of risk management required, demonstrating how quantitative risk assessment translates into practical oversight strategies.")


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_risk_by_category_plotly(df, category_col, score_col):
    """
    Generates an aggregated comparison bar chart of average model risk scores across
    different categories using Plotly.
    """
    if df.empty:
        st.warning("DataFrame is empty, cannot generate plot.")
        return go.Figure() # Return an empty figure

    if category_col not in df.columns:
        raise KeyError(f"Column \'{category_col}\' not found in the DataFrame.")
    if score_col not in df.columns:
        raise KeyError(f"Column \'{score_col}\' not found in the DataFrame.")

    # Ensure category_col is not empty or contains only NaNs before grouping
    if df[category_col].isnull().all():
        st.warning(f"Column \'{category_col}\' contains only missing values. Cannot group by this column.")
        return go.Figure()

    aggregated_df = df.groupby(category_col, observed=False)[score_col].mean().reset_index()
    aggregated_df = aggregated_df.sort_values(by=score_col, ascending=False)

    fig = px.bar(
        aggregated_df,
        x=category_col,
        y=score_col,
        title=f'Average Model Risk Score by {category_col.replace("_", " ").title()}',
        labels={category_col: category_col.replace("_", " ").title(), score_col: f'Average {score_col.replace("_", " ").title()}'},
        color=score_col,
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(xaxis_title=category_col.replace("_", " ").title(), yaxis_title=f'Average {score_col.replace("_", " ").title()}')
    return fig

def plot_risk_heatmap_plotly(df, x_col, y_col, value_col):
    """
    Generates a heatmap showing the average model risk score across two categorical dimensions using Plotly.
    """
    if df.empty:
        st.warning("DataFrame is empty, cannot generate plot.")
        return go.Figure()

    for col in [x_col, y_col, value_col]:
        if col not in df.columns:
            raise KeyError(f"Column \'{col}\' not found in the DataFrame.")

    # Ensure value_col is numeric before attempting mean
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        st.error(f"Column \'{value_col}\' is not numeric. Cannot calculate mean for heatmap.")
        return go.Figure()

    aggregated_data = df.groupby([y_col, x_col], observed=False)[value_col].mean().reset_index()

    # Ensure order for categorical axes
    if pd.api.types.is_categorical_dtype(df[x_col]):
        x_order = df[x_col].cat.categories.tolist()
    else:
        x_order = sorted(df[x_col].unique())

    if pd.api.types.is_categorical_dtype(df[y_col]):
        y_order = df[y_col].cat.categories.tolist()
    else:
        y_order = sorted(df[y_col].unique(), reverse=True) # Reverse for typical heatmap display where higher values are at the top

    pivot_table = aggregated_data.pivot_table(index=y_col, columns=x_col, values=value_col)
    
    # Reindex to ensure consistent order
    pivot_table = pivot_table.reindex(index=y_order, columns=x_order)

    fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns.tolist(),
            y=pivot_table.index.tolist(),
            colorscale='YlGnBu',
            colorbar=dict(title=value_col.replace("_", " ").title())
        ))

    fig.update_layout(
        title=f'Average {value_col.replace("_", " ").title()} by {y_col.replace("_", " ").title()} and {x_col.replace("_", " ").title()}',
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        xaxis_nticks=len(pivot_table.columns),
        yaxis_nticks=len(pivot_table.index)
    )
    fig.update_xaxes(side="top")
    return fig

def plot_risk_relationship_plotly(df, x_col, y_col, hue_col=None):
    """
    Generates a scatter plot to visualize the relationship between two parameters,
    optionally colored by a third categorical parameter using Plotly.
    """
    if df.empty:
        st.warning("DataFrame is empty, cannot generate plot.")
        return go.Figure()

    for col in [x_col, y_col]:
        if col not in df.columns:
            raise KeyError(f"Column \'{col}\' not found in DataFrame.")
    if hue_col is not None and hue_col not in df.columns:
        raise KeyError(f"Column \'{hue_col}\' not found in DataFrame.")

    title = f'Relationship between {x_col.replace("_", " ").title()} and {y_col.replace("_", " ").title()}'
    if hue_col:
        title += f' by {hue_col.replace("_", " ").title()}'
    
    # Handle categorical `hue_col` for consistent coloring
    color_discrete_map = None
    if hue_col and pd.api.types.is_categorical_dtype(df[hue_col]):
        # Create a consistent color map if hue_col is categorical
        categories = df[hue_col].cat.categories.tolist()
        colors = px.colors.qualitative.Plotly # or any other suitable palette
        color_discrete_map = {category: colors[i % len(colors)] for i, category in enumerate(categories)}


    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=hue_col,
        title=title,
        labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
        color_discrete_map=color_discrete_map,
        hover_data=df.columns.tolist() # Show all column data on hover
    )

    fig.update_traces(marker=dict(size=8, opacity=0.8))
    return fig

def perform_sensitivity_analysis(base_model_params, param_to_vary, variation_values, weights, factor_mappings):
    """
    Conducts sensitivity analysis by varying a single model parameter while keeping others constant,
    calculating the resulting model risk scores.
    """
    if param_to_vary not in weights:
        raise KeyError(f"Parameter to vary \'{param_to_vary}\' is not defined in the \'weights\' dictionary, "
                       f"and therefore cannot be analyzed as a factor contributing to the model risk score.")
    if param_to_vary not in factor_mappings:
        raise KeyError(f"Parameter to vary \'{param_to_vary}\' is not defined in the \'factor_mappings\' dictionary, "
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
                try:
                    mapped_value = mapper[factor_value]
                except KeyError as e:
                    st.error(f"Error in sensitivity analysis: Unknown categorical value ' {factor_value}' encountered for factor '{factor}'. "
                             f"Expected one of: {list(mapper.keys())}.")
                    mapped_value = 0 # Assign a default or handle error gracefully
            elif callable(mapper):
                mapped_value = mapper(factor_value)
            else:
                raise TypeError(f"Unsupported mapping type for factor \'{factor}\': {type(mapper)}. "
                                f"Expected dict or callable.")
            
            model_risk_score += weight * mapped_value
        
        results_data.append({param_to_vary: value, 'model_risk_score': model_risk_score})

    return pd.DataFrame(results_data)

def run_page3():
    st.markdown("## 8. Visualizing Model Risk Distribution by Business Impact")
    st.markdown("""
    Understanding how model risk varies across different `business_impact_category` levels is crucial for prioritizing risk management efforts. A bar plot will allow us to visualize the average model risk score for each business impact category, clearly showing which categories pose higher inherent risks. This directly ties back to materiality, where higher business impact generally implies higher risk and necessitates more rigorous management.
    """)

    if 'models_with_guidance_df' not in st.session_state or st.session_state.models_with_guidance_df.empty:
        st.warning("Please go to 'Page 1' and 'Page 2' to generate data and calculate risk scores first.")
        return

    st.subheader("Average Model Risk Score by Business Impact Category")
    fig_bar = plot_risk_by_category_plotly(st.session_state.models_with_guidance_df, 'business_impact_category', 'model_risk_score')
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("The bar plot clearly illustrates a direct correlation between the `business_impact_category` and the `average model_risk_score`. Models categorized with 'Critical' business impact exhibit significantly higher average risk scores, underscoring the importance of materiality in focusing risk management resources where potential adverse consequences are greatest.")

    st.markdown("## 9. Visualizing Model Risk across Complexity and Business Impact (Heatmap)")
    st.markdown("""
    While individual factors contribute to risk, understanding their combined effect is vital. A heatmap provides an excellent way to visualize the average `model_risk_score` across two categorical dimensions simultaneously: `complexity_level` and `business_impact_category`. This allows us to identify specific combinations of model characteristics that result in the highest risk concentrations.
    """)

    st.subheader("Average Model Risk Score Heatmap")
    fig_heatmap = plot_risk_heatmap_plotly(st.session_state.models_with_guidance_df, 'complexity_level', 'business_impact_category', 'model_risk_score')
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("The heatmap reveals distinct patterns, showing that models with 'High' complexity and 'Critical' business impact consistently have the highest average `model_risk_score`. This visualization effectively highlights that model risk is not solely an additive function but can be exacerbated by the interaction of multiple high-risk attributes, guiding more targeted risk mitigation strategies.")

    st.markdown("## 10. Relationship between Input Parameters and Model Risk (Scatter Plot)")
    st.markdown("""
    A scatter plot can help us explore the relationship between a continuous input parameter and the calculated `model_risk_score`. Here, we will visualize how `data_quality_index` influences `model_risk_score`, with points colored by `complexity_level`. We expect to see lower data quality (lower index) generally leading to higher risk scores, and potentially observing how complexity affects this relationship.
    """)

    st.subheader("Model Risk Score vs. Data Quality Index")
    fig_scatter = plot_risk_relationship_plotly(st.session_state.models_with_guidance_df, 'data_quality_index', 'model_risk_score', hue_col='complexity_level')
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("The scatter plot demonstrates the expected inverse relationship: as `data_quality_index` decreases (indicating poorer data quality), the `model_risk_score` tends to increase. Furthermore, models with 'High' complexity often exhibit higher risk scores across similar data quality levels compared to 'Low' or 'Medium' complexity models, reinforcing the multi-factorial nature of model risk.")

    st.markdown("## 11. Sensitivity Analysis: Impact of Complexity on Model Risk")
    st.markdown("""
    Sensitivity analysis allows us to understand how changes in a single input parameter affect the `model_risk_score`, while holding other factors constant. This provides insights into the relative importance and leverage points for risk mitigation.

    We will simulate a hypothetical model and vary its `complexity_level` while keeping its `data_quality_index`, `usage_frequency`, and `business_impact_category` constant. This will show us how the `model_risk_score` responds to changes in complexity.
    """)

    st.sidebar.subheader("Sensitivity Analysis Controls")
    st.sidebar.markdown("**Base Model Parameters:**")

    # Initialize base model parameters in session state
    if "base_complexity_level" not in st.session_state:
        st.session_state.base_complexity_level = 'Medium'
    if "base_data_quality_index" not in st.session_state:
        st.session_state.base_data_quality_index = 75
    if "base_usage_frequency" not in st.session_state:
        st.session_state.base_usage_frequency = 'Medium'
    if "base_business_impact_category" not in st.session_state:
        st.session_state.base_business_impact_category = 'High'

    base_complexity_level = st.sidebar.selectbox(
        "Complexity Level", ('Low', 'Medium', 'High'), key="base_comp_level",
        index=('Low', 'Medium', 'High').index(st.session_state.base_complexity_level)
    )
    base_data_quality_index = st.sidebar.number_input(
        "Data Quality Index (50-100)", 50, 100, st.session_state.base_data_quality_index, key="base_dq_index"
    )
    base_usage_frequency = st.sidebar.selectbox(
        "Usage Frequency", ('Low', 'Medium', 'High'), key="base_usage_freq",
        index=('Low', 'Medium', 'High').index(st.session_state.base_usage_frequency)
    )
    base_business_impact_category = st.sidebar.selectbox(
        "Business Impact Category", ('Low', 'Medium', 'High', 'Critical'), key="base_bi_cat",
        index=('Low', 'Medium', 'High', 'Critical').index(st.session_state.base_business_impact_category)
    )

    st.session_state.base_model_params = {
        'complexity_level': base_complexity_level,
        'data_quality_index': base_data_quality_index,
        'usage_frequency': base_usage_frequency,
        'business_impact_category': base_business_impact_category
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Parameter to Vary:**")
    param_to_vary_option = st.sidebar.selectbox(
        "Select Parameter to Vary",
        ('complexity_level', 'data_quality_index'),
        key="param_to_vary_option"
    )

    variation_values = []
    if param_to_vary_option == 'complexity_level':
        variation_values = st.sidebar.multiselect(
            "Complexity Levels to Test",
            ('Low', 'Medium', 'High'),
            default=('Low', 'Medium', 'High'),
            key="complexity_variation_values"
        )
    elif param_to_vary_option == 'data_quality_index':
        min_dq, max_dq = st.sidebar.slider(
            "Data Quality Index Range",
            50, 100, (50, 100), 5,
            key="dq_variation_range"
        )
        variation_values = list(range(min_dq, max_dq + 1, 5))

    if st.sidebar.button("Run Sensitivity Analysis", key="run_sensitivity_button"):
        if not variation_values:
            st.error("Please select at least one value to vary for the chosen parameter.")
        elif 'wc_weight' not in st.session_state or 'wdq_weight' not in st.session_state or \
             'wuf_weight' not in st.session_state or 'wbi_weight' not in st.session_state:
            st.error("Weights are not defined. Please go to 'Page 2' to define weights.")
        else:
            current_weights = {
                'complexity_level': st.session_state.wc_weight,
                'data_quality_index': st.session_state.wdq_weight,
                'usage_frequency': st.session_state.wuf_weight,
                'business_impact_category': st.session_state.wbi_weight
            }
            try:
                st.session_state.sensitivity_df = perform_sensitivity_analysis(
                    st.session_state.base_model_params,
                    param_to_vary_option,
                    variation_values,
                    current_weights,
                    st.session_state.factor_mappings
                )
            except (KeyError, ValueError, TypeError) as e:
                st.error(f"Error during sensitivity analysis: {e}")
                st.session_state.sensitivity_df = pd.DataFrame() # Clear dataframe on error

    if 'sensitivity_df' in st.session_state and not st.session_state.sensitivity_df.empty:
        if param_to_vary_option == 'complexity_level':
            st.subheader("Sensitivity Analysis: Complexity Level")
            fig_comp_sens = px.line(
                st.session_state.sensitivity_df,
                x='complexity_level',
                y='model_risk_score',
                title='Sensitivity of Model Risk Score to Complexity Level',
                markers=True
            )
            fig_comp_sens.update_layout(xaxis_title='Complexity Level', yaxis_title='Model Risk Score')
            st.plotly_chart(fig_comp_sens, use_container_width=True)
            st.markdown("The sensitivity analysis clearly shows that increasing a model's `complexity_level` (from Low to High) directly leads to a higher `model_risk_score`, assuming all other factors remain constant. This highlights that model complexity is a significant driver of risk, and managing complexity can be an effective strategy for reducing overall model risk.")

        elif param_to_vary_option == 'data_quality_index':
            st.subheader("Sensitivity Analysis: Data Quality Index")
            fig_dq_sens = px.line(
                st.session_state.sensitivity_df,
                x='data_quality_index',
                y='model_risk_score',
                title='Sensitivity of Model Risk Score to Data Quality Index',
                markers=True
            )
            fig_dq_sens.update_layout(xaxis_title='Data Quality Index (Higher is Better)', yaxis_title='Model Risk Score')
            st.plotly_chart(fig_dq_sens, use_container_width=True)
            st.markdown("The sensitivity analysis demonstrates that improving `data_quality_index` (increasing the index value) leads to a reduction in the `model_risk_score`. Conversely, poorer data quality significantly elevates the risk. This underscores that investment in data quality initiatives is a direct and impactful way to mitigate model risk.")
    else:
        st.info("No sensitivity analysis results to display yet. Adjust parameters in the sidebar and click 'Run Sensitivity Analysis'.")

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

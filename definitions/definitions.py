import pandas as pd
import numpy as np

def generate_synthetic_model_data(num_models):
    """
    Generates a Pandas DataFrame of synthetic model attributes for risk analysis.

    Arguments:
    num_models (integer): The number of synthetic models to generate.

    Output:
    Pandas DataFrame with columns: `model_id`, `complexity_level`, `data_quality_index`,
    `usage_frequency`, `business_impact_category`.
    """
    # Input validation for num_models
    if not isinstance(num_models, int):
        raise TypeError("num_models must be an integer.")
    if num_models < 0:
        raise ValueError("num_models must be a non-negative integer.")

    # Define the expected columns
    expected_columns = [
        'model_id',
        'complexity_level',
        'data_quality_index',
        'usage_frequency',
        'business_impact_category'
    ]

    # Handle the case where num_models is 0 by returning an empty DataFrame with correct columns
    if num_models == 0:
        return pd.DataFrame(columns=expected_columns)

    # Define the possible values for categorical columns
    complexity_levels = ['Low', 'Medium', 'High']
    usage_frequencies = ['Low', 'Medium', 'High']
    business_impact_categories = ['Low', 'Medium', 'High', 'Critical']

    # Generate synthetic data for each column
    data = {
        'model_id': np.arange(num_models),
        'complexity_level': np.random.choice(complexity_levels, size=num_models),
        'data_quality_index': np.random.randint(0, 101, size=num_models), # Inclusive of 0, 100
        'usage_frequency': np.random.choice(usage_frequencies, size=num_models),
        'business_impact_category': np.random.choice(business_impact_categories, size=num_models)
    }

    # Create the DataFrame, ensuring column order
    df = pd.DataFrame(data, columns=expected_columns)

    return df

import pandas as pd
import numpy as np

def validate_and_summarize_data(df):
    """
    Performs validation checks and prints summary statistics for the input DataFrame.
    This ensures the integrity and consistency of the dataset by confirming expected
    column names and data types, checking for primary-key uniqueness for `model_id`,
    and asserting no missing values in critical columns, followed by displaying
    descriptive statistics.

    Arguments:
    df (Pandas DataFrame): The DataFrame to be validated and summarized.

    Output:
    Prints validation results (column names check, data types check, primary-key
    uniqueness for `model_id`, missing value assertion for critical columns) and
    summary statistics (e.g., `df.describe()`, value counts for categorical columns).
    """

    # 1. Input Type Validation
    if not isinstance(df, pd.DataFrame):
        print(f"Error: Expected a Pandas DataFrame as input, but got {type(df).__name__}.")
        return

    # Define expected schema based on test fixtures
    expected_columns_list = [
        'model_id', 'complexity_level', 'data_quality_index',
        'usage_frequency', 'business_impact_category'
    ]

    # Expected dtypes as strings for comparison (considering how pandas represents them)
    # The categories for categorical dtypes are not strictly checked by name match,
    # but the 'category' type itself is.
    # Note: `astype(int)` results in `int64`, `astype(str)` results in `object`.
    # `np.nan` in `int64` column converts it to `float64`.
    expected_dtype_names = {
        'model_id': ['object'],
        'complexity_level': ['category'],
        'data_quality_index': ['int64'],
        'usage_frequency': ['category'],
        'business_impact_category': ['category']
    }
    
    critical_columns_for_na_check = expected_columns_list # All expected columns are critical

    print("--- DataFrame Validation ---")

    # 2. Column Name Validation
    df_columns = set(df.columns)
    expected_columns_set = set(expected_columns_list)

    missing_expected_columns = expected_columns_set - df_columns
    unexpected_columns = df_columns - expected_columns_set

    if missing_expected_columns:
        print(f"Missing expected columns: {sorted(list(missing_expected_columns))}")
    else:
        print("All expected columns are present.")

    if unexpected_columns:
        print(f"Unexpected columns found: {sorted(list(unexpected_columns))}")
    
    # Check if a critical column is missing. If so, subsequent checks on it won't run.
    # This also helps to ensure `df.empty` check isn't the *only* thing said if columns are wrong.

    # 3. Data Type Validation (only for present expected columns)
    current_all_dtypes_match = True
    for col, expected_names in expected_dtype_names.items():
        if col in df.columns: # Only check if column exists
            actual_dtype_name = str(df[col].dtype) # Convert to string representation for comparison
            if not any(expected_name == actual_dtype_name for expected_name in expected_names):
                print(f"Mismatch in data type for column: {col} (expected {', '.join(expected_names)}, got {actual_dtype_name})")
                current_all_dtypes_match = False
    
    if current_all_dtypes_match:
        print("All column data types are as expected.")

    # 4. Primary Key Uniqueness for 'model_id'
    if 'model_id' in df.columns:
        if df.empty:
            print("model_id is unique. (DataFrame is empty)") # Vacuously true for no rows
        elif df['model_id'].duplicated().any():
            print("Duplicate model_id values found.")
        else:
            print("model_id is unique.")
    # else: 'model_id' is missing, message already printed by column validation.

    # 5. Missing Values Assertion in Critical Columns
    found_missing_in_critical = []
    for col in critical_columns_for_na_check:
        if col in df.columns: # Only check if column exists
            if df[col].isnull().any():
                found_missing_in_critical.append(col)
    
    if found_missing_in_critical:
        print(f"Missing values found in critical columns: {sorted(found_missing_in_critical)}")
    else:
        print("No missing values found in critical columns.")

    print("\n--- Summary Statistics for DataFrame ---")

    if df.empty:
        print("DataFrame is empty.")
    
    # Description for numerical columns (will still print 'count 0.0' for empty DF)
    numerical_df = df.select_dtypes(include=np.number)
    if not numerical_df.empty or (df.empty and not numerical_df.columns.empty):
        # This condition ensures we print describe if there are numerical columns,
        # even if the dataframe is empty of rows.
        print("\nDescription for numerical columns:")
        print(numerical_df.describe().to_string())
    elif numerical_df.empty and df.empty:
        print("\nNo numerical columns found for description.") # If empty DF and no numerical columns
    
    # Value counts for categorical columns
    categorical_df = df.select_dtypes(include='category')
    if not categorical_df.empty: # Check if there ARE categorical columns
        print("\nValue counts for categorical columns:")
        for col in categorical_df.columns:
            print(f"\n--- {col} ---")
            print(df[col].value_counts(dropna=False).to_string())
    elif categorical_df.empty and df.empty:
        print("\nNo categorical columns found.")

import pandas as pd
import numpy as np

def calculate_model_risk_score(df, weights, factor_mappings):
    """Computes a model risk score for each model in the DataFrame based on a weighted sum methodology.

    Each model characteristic is assigned a numerical score, and these scores are combined using
    predefined weights to calculate a composite score reflecting the overall risk profile.

    Arguments:
        df (Pandas DataFrame): The input DataFrame containing model attributes.
        weights (dictionary): A dictionary of weights for each risk factor, summing to 1.
        factor_mappings (dictionary): A dictionary defining how categorical levels map to numerical scores
                                      and how continuous values are transformed.

    Output:
        Pandas DataFrame with an additional `model_risk_score` column.
    """

    # Validate: Weights must sum to 1.0 (allowing for floating point inaccuracies)
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"Weights must sum to 1.0. Current sum: {sum(weights.values())}")

    # Create a copy of the DataFrame to add the new column without modifying the original input
    df_output = df.copy()

    # Initialize the model_risk_score column to zero
    df_output['model_risk_score'] = 0.0

    # Iterate through each factor defined in the weights to calculate its contribution
    for factor, weight in weights.items():
        # Ensure the factor column exists in the DataFrame
        if factor not in df_output.columns:
            raise KeyError(f"Missing required column: '{factor}' in the input DataFrame.")

        # Get the mapping configuration for the current factor
        mapping = factor_mappings.get(factor)
        if mapping is None:
            raise ValueError(f"No mapping defined in 'factor_mappings' for factor: '{factor}'.")

        factor_scores = None
        if isinstance(mapping, dict):
            # Categorical factor: Map DataFrame values using the dictionary.
            # Using .apply() with a lambda to ensure a KeyError is raised for unknown categorical values,
            # aligning with test case expectations.
            try:
                factor_scores = df_output[factor].apply(lambda x: mapping[x])
            except KeyError as e:
                # Re-raise with more context for clarity in case of unknown categorical values
                raise KeyError(f"Unknown categorical value '{e.args[0]}' encountered for factor '{factor}'. "
                               f"Expected one of: {list(mapping.keys())}.") from e
        elif callable(mapping):
            # Continuous factor: Apply the transformation function
            factor_scores = df_output[factor].apply(mapping)
        else:
            # Handle unsupported mapping types
            raise TypeError(f"Unsupported mapping type for factor '{factor}'. "
                            "Mapping must be a dictionary for categorical values or a callable for continuous values.")

        # Add the weighted score contribution to the total model_risk_score
        df_output['model_risk_score'] += factor_scores * weight

    return df_output

import pandas as pd
import numpy as np

def provide_materiality_guidance(df, score_thresholds):
    """
    Assigns a risk management guidance level to each model based on its calculated model risk score and predefined thresholds.
    """
    if df.empty:
        # For an empty DataFrame, ensure the 'management_guidance' column is added with the correct string dtype.
        df['management_guidance'] = pd.Series([], dtype=str)
        return df

    # Define conditions for different guidance levels based on score_thresholds.
    # Accessing 'model_risk_score' will raise KeyError if the column does not exist.
    # Operations on 'df' will raise AttributeError if 'df' is not a DataFrame.
    conditions = [
        df['model_risk_score'] <= score_thresholds['Standard Oversight'],
        (df['model_risk_score'] > score_thresholds['Standard Oversight']) & (df['model_risk_score'] <= score_thresholds['Enhanced Scrutiny']),
        df['model_risk_score'] > score_thresholds['Enhanced Scrutiny']
    ]

    # Define the corresponding guidance levels.
    choices = [
        'Standard Oversight',
        'Enhanced Scrutiny',
        'Rigorous Management'
    ]

    # Use numpy.select for efficient, vectorized application of conditions.
    df['management_guidance'] = np.select(conditions, choices)

    # Ensure the 'management_guidance' column has a string data type, as np.select might default to 'object'.
    df['management_guidance'] = df['management_guidance'].astype(str)

    return df

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_risk_by_category(df, category_col, score_col, plot_type):
    """
    Generates an aggregated comparison plot (e.g., bar chart) of model risk scores across
    different categories. This visualization helps in understanding how model risk varies
    across different categorical levels, such as business impact or usage frequency.

    Arguments:
        df (Pandas DataFrame): The input DataFrame.
        category_col (str): The name of the categorical column for grouping.
        score_col (str): The name of the model risk score column.
        plot_type (str): The type of plot to generate ('bar' or 'heatmap').

    Returns:
        matplotlib.axes.Axes: A Matplotlib Axes object with the generated plot.
    """

    # 1. Input Validation
    if category_col not in df.columns:
        raise KeyError(f"Column '{category_col}' not found in the DataFrame.")
    if score_col not in df.columns:
        raise KeyError(f"Column '{score_col}' not found in the DataFrame.")

    valid_plot_types = ['bar', 'heatmap']
    if plot_type not in valid_plot_types:
        raise ValueError(f"plot_type must be '{' or '.join(valid_plot_types)}'")

    # 2. Data Aggregation
    # Group by category_col and calculate the mean of score_col.
    # Use .reset_index() to convert the Series back to a DataFrame for easier plotting.
    # This handles empty DF gracefully, resulting in an empty DataFrame if the input is empty.
    # observed=False is included for robust handling of categorical dtypes, though
    # for string columns, it might not change behavior significantly.
    aggregated_df = df.groupby(category_col, observed=False)[score_col].mean().reset_index()

    # Create a figure and an axes object
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Plot Generation
    if plot_type == 'bar':
        # seaborn.barplot can handle an empty aggregated_df, it will just show an empty plot
        sns.barplot(x=category_col, y=score_col, data=aggregated_df, ax=ax, palette="viridis")
        ax.set_title(f'Average Model Risk Score by {category_col}')
        ax.set_xlabel(category_col)
        ax.set_ylabel(f'Average {score_col}')
        # Rotate x-axis labels for readability if there is actual data
        if not aggregated_df.empty:
            ax.tick_params(axis='x', rotation=45)
    elif plot_type == 'heatmap':
        # For heatmap, reshape data to have categories as index.
        # This will create a single-column heatmap where categories are on the y-axis.
        # sns.heatmap can generally handle an empty DataFrame (e.g., it will draw an empty grid).
        heatmap_data = aggregated_df.set_index(category_col)
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".1f", cbar=True, linewidths=.5, ax=ax)
        ax.set_title(f'Model Risk Heatmap by {category_col}')
        ax.set_xlabel('') # Clear x-label as categories are on y-axis
        ax.set_ylabel(category_col)

    plt.tight_layout()
    # plt.show() # Uncomment for local testing if needed

    return ax

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.axes as axes

def plot_risk_heatmap(df, x_col, y_col, value_col) -> axes.Axes:
    """
    Generates a heatmap showing the average model risk score across two categorical dimensions.
    This visualization helps to identify specific combinations of model characteristics that result
    in the highest risk concentrations.

    Arguments:
        df (Pandas DataFrame): The input DataFrame.
        x_col (string): The column for the x-axis of the heatmap.
        y_col (string): The column for the y-axis of the heatmap.
        value_col (string): The column for the aggregation value, typically 'model_risk_score'.

    Output:
        matplotlib.axes.Axes: A Matplotlib Axes object containing the heatmap.
    """
    # Validate that all required columns exist in the DataFrame
    for col in [x_col, y_col, value_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame.")

    # Aggregate the data by grouping x_col and y_col and calculating the mean of value_col.
    # This step might raise a TypeError if value_col contains non-numeric data that cannot be averaged.
    try:
        # Ensure the value_col is numeric for aggregation.
        # .mean() will raise TypeError if the underlying data is truly non-numeric (e.g., strings).
        aggregated_data = df.groupby([y_col, x_col])[value_col].mean().reset_index()
    except TypeError as e:
        raise TypeError(f"Cannot calculate mean on non-numeric column '{value_col}'. "
                        "Ensure 'value_col' contains numeric data. Error: {e}") from e

    # Pivot the aggregated data to create a matrix suitable for a heatmap.
    # The index will be y_col, columns will be x_col, and values will be the aggregated risk score.
    # Missing combinations will result in NaN, which seaborn.heatmap handles gracefully by not drawing that cell.
    pivot_table = aggregated_data.pivot(index=y_col, columns=x_col, values=value_col)

    # Create a Matplotlib figure and axes for the heatmap
    fig, ax = plt.subplots(figsize=(10, 7)) 
    
    # Generate the heatmap using seaborn
    # annot=True to display the value in each cell
    # fmt=".2f" to format annotations to two decimal places
    # cmap="YlGnBu" for a sequential color map suitable for risk scores
    # linewidths=.5 adds lines between cells for better visual separation
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, ax=ax)

    # Set titles and labels for better readability
    ax.set_title(
        f'Average {value_col.replace("_", " ").title()} by '
        f'{y_col.replace("_", " ").title()} and {x_col.replace("_", " ").title()}',
        fontsize=14
    )
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()

    return ax

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_risk_relationship(df, x_col, y_col, hue_col):
    """
    Generates a scatter plot to visualize the relationship between two parameters,
    optionally colored by a third categorical parameter.
    
    Arguments:
    df (Pandas DataFrame): The input DataFrame.
    x_col (string): The column for the x-axis of the scatter plot.
    y_col (string): The column for the y-axis of the scatter plot.
    hue_col (string, optional): An optional column for color encoding in the plot.
    
    Output:
    A visualization object (e.g., Matplotlib Axes object).
    """

    # Validate DataFrame type
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a Pandas DataFrame.")

    # Validate column existence
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    if hue_col is not None and hue_col not in df.columns:
        raise KeyError(f"Column '{hue_col}' not found in DataFrame.")

    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate the scatter plot using Seaborn
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)

    # Set plot title and labels
    title = f'Relationship between {x_col} and {y_col}'
    if hue_col:
        title += f' by {hue_col}'
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    # Add legend if hue_col is provided
    if hue_col:
        ax.legend(title=hue_col)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    return ax

import pandas as pd

def perform_sensitivity_analysis(base_model_params, param_to_vary, variation_values, weights, factor_mappings):
    """
    Conducts sensitivity analysis by varying a single model parameter while keeping others constant,
    calculating the resulting model risk scores.

    Arguments:
    base_model_params (dictionary): A dictionary of fixed parameter values for a hypothetical model.
    param_to_vary (string): The name of the parameter to vary during the analysis.
    variation_values (list): A list of values to test for the parameter being varied.
    weights (dictionary): The dictionary of weights for each factor, used in risk score calculation.
    factor_mappings (dictionary): The dictionary defining factor to numerical score mappings.

    Output:
    Pandas DataFrame showing how `model_risk_score` changes with `param_to_vary`.
    """
    
    # Validate param_to_vary: it must be a factor recognized by the scoring model.
    # It must have a weight and a mapping defined to be part of the risk score calculation.
    if param_to_vary not in weights:
        raise KeyError(f"Parameter to vary '{param_to_vary}' is not defined in the 'weights' dictionary, "
                       f"and therefore cannot be analyzed as a factor contributing to the model risk score.")
    if param_to_vary not in factor_mappings:
        raise KeyError(f"Parameter to vary '{param_to_vary}' is not defined in the 'factor_mappings' dictionary, "
                       f"and therefore cannot be analyzed as a factor contributing to the model risk score.")

    results_data = []

    # Handle edge case: empty variation_values list
    if not variation_values:
        # Return an empty DataFrame with the specified column names
        return pd.DataFrame(columns=[param_to_vary, 'model_risk_score'])

    # Iterate through each value in the variation_values list
    for value in variation_values:
        # Create a mutable copy of the base parameters
        current_params = base_model_params.copy()
        # Update the parameter being varied with the current value
        current_params[param_to_vary] = value

        model_risk_score = 0.0

        # Calculate the model risk score based on all weighted factors
        # The iteration covers all factors for which a weight is defined.
        for factor, weight in weights.items():
            # Retrieve the current value for the factor from the (potentially varied) parameters
            # This can raise KeyError if a factor listed in weights is missing from current_params
            # (which implies base_model_params was incomplete, or param_to_vary was not handled correctly)
            factor_value = current_params[factor]
            
            # Retrieve the mapping definition for the current factor
            # This can raise KeyError if a factor listed in weights is missing from factor_mappings (Test Case 5)
            mapper = factor_mappings[factor]

            mapped_value = None
            if isinstance(mapper, dict):
                # If the mapper is a dictionary, it's a categorical mapping
                # This can raise KeyError if factor_value is not a key in the mapper dictionary
                mapped_value = mapper[factor_value]
            elif callable(mapper):
                # If the mapper is a callable, it's a function-based mapping
                mapped_value = mapper(factor_value)
            else:
                # This scenario should not occur given the problem description and test data structure,
                # but it serves as a robust check for unexpected mapping types.
                raise TypeError(f"Unsupported mapping type for factor '{factor}': {type(mapper)}. "
                                f"Expected dict or callable.")
            
            # Add the weighted and mapped score for this factor to the total model risk score
            model_risk_score += weight * mapped_value
        
        # Store the varied parameter's value and the calculated risk score
        results_data.append({param_to_vary: value, 'model_risk_score': model_risk_score})

    # Convert the collected results into a Pandas DataFrame
    return pd.DataFrame(results_data)
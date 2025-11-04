import pytest
import pandas as pd
import numpy as np

# Keep the definition_33ba50f2009b461f981ec711867c76d9 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_33ba50f2009b461f981ec711867c76d9 import validate_and_summarize_data

@pytest.fixture
def base_valid_df():
    """Returns a DataFrame that should pass all validation checks."""
    data = {
        'model_id': ['model_001', 'model_002', 'model_003', 'model_004'],
        'complexity_level': ['Low', 'Medium', 'High', 'Low'],
        'data_quality_index': [85, 90, 75, 95],
        'usage_frequency': ['Low', 'Medium', 'High', 'Medium'],
        'business_impact_category': ['Low', 'Medium', 'High', 'Critical']
    }
    df = pd.DataFrame(data)
    # Ensure dtypes are explicit for strict checking
    df['model_id'] = df['model_id'].astype(str)
    df['complexity_level'] = pd.Categorical(df['complexity_level'], categories=['Low', 'Medium', 'High'], ordered=True)
    df['data_quality_index'] = df['data_quality_index'].astype(int)
    df['usage_frequency'] = pd.Categorical(df['usage_frequency'], categories=['Low', 'Medium', 'High'], ordered=True)
    df['business_impact_category'] = pd.Categorical(df['business_impact_category'], categories=['Low', 'Medium', 'High', 'Critical'], ordered=True)
    return df

@pytest.fixture
def expected_columns():
    return [
        'model_id', 'complexity_level', 'data_quality_index',
        'usage_frequency', 'business_impact_category'
    ]

def test_valid_dataframe(base_valid_df, capfd):
    """
    Test case 1: Valid DataFrame with expected columns, dtypes, unique model_id, and no missing values.
    Expects successful validation messages and summary statistics.
    """
    validate_and_summarize_data(base_valid_df)
    captured = capfd.readouterr()
    output = captured.out

    assert "All expected columns are present." in output
    assert "All column data types are as expected." in output
    assert "model_id is unique." in output
    assert "No missing values found in critical columns." in output
    assert "Summary Statistics for DataFrame:" in output
    assert "Description for numerical columns:" in output
    assert "Value counts for categorical columns:" in output

def test_empty_dataframe(capfd, expected_columns):
    """
    Test case 2: Empty DataFrame with correct column structure.
    Expects validation messages that handle an empty state gracefully, and empty summary.
    """
    empty_df = pd.DataFrame(columns=expected_columns)
    # Explicitly set dtypes for columns if function is strict, even if empty
    empty_df['model_id'] = empty_df['model_id'].astype(str)
    empty_df['complexity_level'] = pd.Categorical(empty_df['complexity_level'], categories=['Low', 'Medium', 'High'], ordered=True)
    empty_df['data_quality_index'] = empty_df['data_quality_index'].astype(int)
    empty_df['usage_frequency'] = pd.Categorical(empty_df['usage_frequency'], categories=['Low', 'Medium', 'High'], ordered=True)
    empty_df['business_impact_category'] = pd.Categorical(empty_df['business_impact_category'], categories=['Low', 'Medium', 'High', 'Critical'], ordered=True)

    validate_and_summarize_data(empty_df)
    captured = capfd.readouterr()
    output = captured.out

    assert "All expected columns are present." in output
    assert "All column data types are as expected." in output
    assert "model_id is unique." in output # Vacuously true for no rows
    assert "No missing values found in critical columns." in output # Vacuously true for no rows
    assert "Summary Statistics for DataFrame:" in output
    assert "DataFrame is empty." in output # A specific message for empty DF
    assert "count    0.0" in output # From df.describe() for numerical column

def test_dataframe_with_issues(base_valid_df, capfd):
    """
    Test case 3: DataFrame with missing values, non-unique model_id, and incorrect data types.
    Expects specific error messages for each issue.
    """
    df_with_issues = base_valid_df.copy()
    
    # Introduce missing value in a critical column (data_quality_index)
    df_with_issues.loc[0, 'data_quality_index'] = np.nan # This will change column dtype to float64
    
    # Introduce non-unique model_id
    df_with_issues.loc[1, 'model_id'] = 'model_001'
    
    # Introduce incorrect data type for complexity_level (expecting category, provide int)
    df_with_issues['complexity_level'] = df_with_issues['complexity_level'].map({'Low': 1, 'Medium': 2, 'High': 3}).astype(int)

    validate_and_summarize_data(df_with_issues)
    captured = capfd.readouterr()
    output = captured.out

    assert "Missing values found in critical columns: ['data_quality_index']" in output
    assert "Duplicate model_id values found." in output
    # Expected int, got float due to NaN
    assert "Mismatch in data type for column: data_quality_index (expected int, got float)" in output or \
           "Mismatch in data type for column: data_quality_index (expected integer, got floating)" in output
    # Expected category, got int
    assert "Mismatch in data type for column: complexity_level (expected category, got int)" in output or \
           "Mismatch in data type for column: complexity_level (expected category, got integer)" in output
    
    assert "All expected columns are present." in output # Column names are still correct
    assert "Summary Statistics for DataFrame:" in output # Should still try to summarize remaining data

def test_dataframe_with_incorrect_columns(base_valid_df, capfd):
    """
    Test case 4: DataFrame with missing expected columns and unexpected additional columns.
    Expects specific messages about column name discrepancies.
    """
    df_bad_cols = base_valid_df.drop(columns=['model_id', 'usage_frequency']) # Missing two expected columns
    df_bad_cols['extra_col_1'] = ['A', 'B', 'C', 'D'] # Add unexpected column
    df_bad_cols['extra_col_2'] = [10, 20, 30, 40]

    validate_and_summarize_data(df_bad_cols)
    captured = capfd.readouterr()
    output = captured.out

    assert "Missing expected columns: ['model_id', 'usage_frequency']" in output or \
           "Missing expected column: model_id" in output and "Missing expected column: usage_frequency" in output
    assert "Unexpected columns found: ['extra_col_1', 'extra_col_2']" in output or \
           "Unexpected column found: extra_col_1" in output and "Unexpected column found: extra_col_2" in output
    
    # Ensure validation for unique model_id and missing values in those columns is not run
    # or reports as not applicable due to missing column.
    assert "model_id is unique." not in output # Because 'model_id' column is missing
    assert "No missing values found in critical columns." not in output # If critical columns are missing
    assert "Summary Statistics for DataFrame:" in output # Should still try to summarize remaining data

@pytest.mark.parametrize("invalid_input, expected_exception", [
    (None, TypeError),
    (123, TypeError),
    ("not a dataframe", TypeError),
    ([1, 2, 3], TypeError),
    ({'a': 1, 'b': 2}, TypeError),
])
def test_non_dataframe_input(invalid_input, expected_exception):
    """
    Test case 5: Non-DataFrame input.
    Expects a TypeError, as the function explicitly takes a DataFrame.
    """
    with pytest.raises(expected_exception) as excinfo:
        validate_and_summarize_data(invalid_input)
    # Check for a specific message if the function's first line validates input type
    assert "Expected a Pandas DataFrame as input" in str(excinfo.value) or \
           "must be a pandas.DataFrame" in str(excinfo.value)
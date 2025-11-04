import pandas as pd
import pytest

# definition_224883d897d749049fa443c457bfce76 block START
from definition_224883d897d749049fa443c457bfce76 import provide_materiality_guidance
# definition_224883d897d749049fa443c457bfce76 block END

# Define the score thresholds as per the notebook specification:
# - model_risk_score <= 3: 'Standard Oversight'
# - 3 < model_risk_score <= 6: 'Enhanced Scrutiny'
# - model_risk_score > 6: 'Rigorous Management'
# This dictionary represents the upper bounds for each category.
# Anything above the last explicit bound (6.0) falls into 'Rigorous Management'.
DEFAULT_THRESHOLDS = {
    'Standard Oversight': 3.0,
    'Enhanced Scrutiny': 6.0,
}

@pytest.mark.parametrize("df_input, expected_output, expected_exception", [
    # Test Case 1: Standard functionality with mixed scores
    # Covers various scores falling into each of the three guidance categories.
    (
        pd.DataFrame({'model_risk_score': [1.5, 4.0, 7.0, 2.9, 5.5, 8.1]}),
        pd.DataFrame({
            'model_risk_score': [1.5, 4.0, 7.0, 2.9, 5.5, 8.1],
            'management_guidance': ['Standard Oversight', 'Enhanced Scrutiny', 'Rigorous Management',
                                    'Standard Oversight', 'Enhanced Scrutiny', 'Rigorous Management']
        }),
        None
    ),
    # Test Case 2: Edge cases - Scores exactly at thresholds or just beyond.
    # Checks behavior at 3.0, 3.0001, 6.0, 6.0001 to ensure correct boundary handling.
    (
        pd.DataFrame({'model_risk_score': [3.0, 3.0001, 6.0, 6.0001, 0.0, 10.0]}),
        pd.DataFrame({
            'model_risk_score': [3.0, 3.0001, 6.0, 6.0001, 0.0, 10.0],
            'management_guidance': ['Standard Oversight', 'Enhanced Scrutiny', 'Enhanced Scrutiny',
                                    'Rigorous Management', 'Standard Oversight', 'Rigorous Management']
        }),
        None
    ),
    # Test Case 3: Empty DataFrame input.
    # The function should handle an empty DataFrame gracefully, returning an empty DataFrame with the new column.
    (
        pd.DataFrame({'model_risk_score': []}, dtype=float),
        pd.DataFrame({
            'model_risk_score': pd.Series([], dtype=float),
            'management_guidance': pd.Series([], dtype=str)
        }),
        None
    ),
    # Test Case 4: DataFrame missing the 'model_risk_score' column.
    # Expects a KeyError as the function tries to access a non-existent column.
    (
        pd.DataFrame({'other_column': [1, 2, 3]}),
        None, # No expected output DataFrame for an error case
        KeyError # Expected exception type
    ),
    # Test Case 5: Invalid 'df' input type (not a Pandas DataFrame).
    # Expects an AttributeError when attempting DataFrame-specific operations (e.g., column access).
    (
        [1, 2, 3], # Input is a list, not a DataFrame
        None,
        AttributeError # Expected exception type
    ),
])
def test_provide_materiality_guidance(df_input, expected_output, expected_exception):
    # Create a copy of the input DataFrame to ensure the original is not modified,
    # especially important for mutable inputs in parametrize.
    df_test_input = df_input.copy() if isinstance(df_input, pd.DataFrame) else df_input

    if expected_exception:
        with pytest.raises(expected_exception):
            provide_materiality_guidance(df_test_input, DEFAULT_THRESHOLDS)
    else:
        result_df = provide_materiality_guidance(df_test_input, DEFAULT_THRESHOLDS)
        # Use pandas.testing.assert_frame_equal for robust DataFrame comparison.
        # check_dtype=True ensures column data types match.
        pd.testing.assert_frame_equal(result_df, expected_output, check_dtype=True)
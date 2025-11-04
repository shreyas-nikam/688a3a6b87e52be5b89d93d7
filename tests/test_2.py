import pytest
import pandas as pd
import numpy as np

from definition_e57aa8e747b64c22a2449f08fa27c474 import calculate_model_risk_score

# Helper data based on the notebook specification for standard weights and mappings
_STANDARD_WEIGHTS = {
    'complexity_level': 0.2,
    'data_quality_index': 0.2,
    'usage_frequency': 0.1,
    'business_impact_category': 0.5
}

_FACTOR_MAPPINGS = {
    'complexity_level': {'Low': 1, 'Medium': 3, 'High': 5},
    'data_quality_index': lambda x: (100 - x) / 10,
    'usage_frequency': {'Low': 1, 'Medium': 3, 'High': 5},
    'business_impact_category': {'Low': 1, 'Medium': 3, 'High': 5, 'Critical': 10}
}

@pytest.mark.parametrize(
    "df_input, weights_input, factor_mappings_input, expected_output_or_exception",
    [
        # Test Case 1: Basic functionality with a single model
        # Covers expected calculation for a typical scenario.
        (
            pd.DataFrame([
                {'model_id': 'M1', 'complexity_level': 'Medium', 'data_quality_index': 70, 'usage_frequency': 'Medium', 'business_impact_category': 'High'}
            ]),
            _STANDARD_WEIGHTS,
            _FACTOR_MAPPINGS,
            pd.Series([4.0], name='model_risk_score') # (0.2*3) + (0.2*3) + (0.1*3) + (0.5*5) = 0.6 + 0.6 + 0.3 + 2.5 = 4.0
        ),
        # Test Case 2: Multiple models, including minimum and maximum valid risk scores within spec
        # Covers multiple rows and boundary values for data_quality_index (50-100).
        (
            pd.DataFrame([
                # Minimum risk model (Low, DQ=100, Low, Low)
                {'model_id': 'M2', 'complexity_level': 'Low', 'data_quality_index': 100, 'usage_frequency': 'Low', 'business_impact_category': 'Low'},
                # Maximum risk model (High, DQ=50, High, Critical)
                {'model_id': 'M3', 'complexity_level': 'High', 'data_quality_index': 50, 'usage_frequency': 'High', 'business_impact_category': 'Critical'},
                # Another typical model
                {'model_id': 'M4', 'complexity_level': 'Low', 'data_quality_index': 80, 'usage_frequency': 'High', 'business_impact_category': 'Medium'}
            ]),
            _STANDARD_WEIGHTS,
            _FACTOR_MAPPINGS,
            pd.Series([
                (0.2*1 + 0.2*0 + 0.1*1 + 0.5*1), # M2: 0.2 + 0 + 0.1 + 0.5 = 0.8
                (0.2*5 + 0.2*5 + 0.1*5 + 0.5*10),# M3: 1.0 + 1.0 + 0.5 + 5.0 = 7.5
                (0.2*1 + 0.2*2 + 0.1*5 + 0.5*3)  # M4: 0.2 + 0.4 + 0.5 + 1.5 = 2.6
            ], name='model_risk_score')
        ),
        # Test Case 3: Edge Case - Invalid DataFrame (missing a required column)
        # Expects a KeyError when trying to access a non-existent column.
        (
            pd.DataFrame([
                {'model_id': 'M5', 'complexity_level': 'Medium', 'data_quality_index': 70, 'usage_frequency': 'Medium'}
            ]),
            _STANDARD_WEIGHTS,
            _FACTOR_MAPPINGS,
            KeyError
        ),
        # Test Case 4: Edge Case - Invalid weights (does not sum to 1)
        # The specification states weights 'summing to 1', implying this should be validated.
        (
            pd.DataFrame([
                {'model_id': 'M6', 'complexity_level': 'Medium', 'data_quality_index': 70, 'usage_frequency': 'Medium', 'business_impact_category': 'High'}
            ]),
            {'complexity_level': 0.5, 'data_quality_index': 0.2, 'usage_frequency': 0.1, 'business_impact_category': 0.5}, # Sums to 1.3
            _FACTOR_MAPPINGS,
            ValueError # Assuming the function validates weights sum to 1
        ),
        # Test Case 5: Edge Case - Invalid DataFrame (unknown categorical value)
        # Expects a KeyError when an unknown category is encountered in factor_mappings.
        (
            pd.DataFrame([
                {'model_id': 'M7', 'complexity_level': 'Invalid', 'data_quality_index': 70, 'usage_frequency': 'Medium', 'business_impact_category': 'High'}
            ]),
            _STANDARD_WEIGHTS,
            _FACTOR_MAPPINGS,
            KeyError
        ),
    ]
)
def test_calculate_model_risk_score(df_input, weights_input, factor_mappings_input, expected_output_or_exception):
    if isinstance(expected_output_or_exception, type) and issubclass(expected_output_or_exception, Exception):
        # Test for expected exceptions
        with pytest.raises(expected_output_or_exception):
            calculate_model_risk_score(df_input, weights_input, factor_mappings_input)
    else:
        # Test for successful computation
        # Create a copy of the input DataFrame to ensure the original is not modified by the function
        df_copy = df_input.copy()
        result_df = calculate_model_risk_score(df_copy, weights_input, factor_mappings_input)

        # Assert that the 'model_risk_score' column exists in the output DataFrame
        assert 'model_risk_score' in result_df.columns

        # Assert that the new 'model_risk_score' column is correctly calculated
        # Using pandas.testing.assert_series_equal for robust Series comparison, allowing for float precision.
        pd.testing.assert_series_equal(
            result_df['model_risk_score'],
            expected_output_or_exception,
            check_exact=False, # Allow for floating point inaccuracies
            atol=1e-6          # Absolute tolerance for comparison
        )

        # Optionally, verify that other columns in the DataFrame remain unchanged
        # This checks that the function only adds the new column and doesn't alter existing data.
        original_cols_in_result = result_df.drop(columns=['model_risk_score'], errors='ignore')
        pd.testing.assert_frame_equal(original_cols_in_result, df_input, check_dtype=True)
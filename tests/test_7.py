import pandas as pd
import pytest
from definition_96063fcb117b49c8ab0dc5bca8dc5e10 import perform_sensitivity_analysis

# Constants for test data (derived from notebook specification)
WEIGHTS = {
    'complexity_level': 0.2,
    'data_quality_index': 0.2,
    'usage_frequency': 0.1,
    'business_impact_category': 0.5
}

FACTOR_MAPPINGS = {
    'complexity_level': {'Low': 1, 'Medium': 3, 'High': 5},
    'data_quality_index': lambda val: (100 - val) / 10, # Function for continuous mapping
    'usage_frequency': {'Low': 1, 'Medium': 3, 'High': 5},
    'business_impact_category': {'Low': 1, 'Medium': 3, 'High': 5, 'Critical': 10}
}

# Helper function to calculate an expected model risk score for a single set of parameters.
# This mimics the scoring logic described in the notebook specification.
def _calculate_expected_risk_score(params, weights, factor_mappings):
    score = 0.0
    
    # Map complexity_level
    complexity_score = factor_mappings['complexity_level'][params['complexity_level']]
    score += weights['complexity_level'] * complexity_score
    
    # Map data_quality_index
    data_quality_score = factor_mappings['data_quality_index'](params['data_quality_index'])
    score += weights['data_quality_index'] * data_quality_score
    
    # Map usage_frequency
    usage_frequency_score = factor_mappings['usage_frequency'][params['usage_frequency']]
    score += weights['usage_frequency'] * usage_frequency_score
    
    # Map business_impact_category
    business_impact_score = factor_mappings['business_impact_category'][params['business_impact_category']]
    score += weights['business_impact_category'] * business_impact_score
    
    return score

@pytest.mark.parametrize(
    "base_model_params, param_to_vary, variation_values, weights, factor_mappings, expected_output, expected_exception",
    [
        # Test Case 1: Varying a categorical parameter (complexity_level)
        # Expected functionality: correctly calculates risk scores for different complexity levels.
        (
            {
                'complexity_level': 'Medium',  # This will be overridden by variation_values
                'data_quality_index': 80,
                'usage_frequency': 'Medium',
                'business_impact_category': 'High'
            },
            'complexity_level',
            ['Low', 'Medium', 'High'],
            WEIGHTS,
            FACTOR_MAPPINGS,
            pd.DataFrame({
                'complexity_level': ['Low', 'Medium', 'High'],
                'model_risk_score': [
                    _calculate_expected_risk_score({'complexity_level': 'Low', 'data_quality_index': 80, 'usage_frequency': 'Medium', 'business_impact_category': 'High'}, WEIGHTS, FACTOR_MAPPINGS),
                    _calculate_expected_risk_score({'complexity_level': 'Medium', 'data_quality_index': 80, 'usage_frequency': 'Medium', 'business_impact_category': 'High'}, WEIGHTS, FACTOR_MAPPINGS),
                    _calculate_expected_risk_score({'complexity_level': 'High', 'data_quality_index': 80, 'usage_frequency': 'Medium', 'business_impact_category': 'High'}, WEIGHTS, FACTOR_MAPPINGS)
                ]
            }),
            None
        ),
        # Test Case 2: Varying a numerical parameter (data_quality_index)
        # Expected functionality: correctly calculates risk scores for different data quality indices.
        (
            {
                'complexity_level': 'Medium',
                'data_quality_index': 80,  # This will be overridden by variation_values
                'usage_frequency': 'Medium',
                'business_impact_category': 'High'
            },
            'data_quality_index',
            [50, 75, 100],
            WEIGHTS,
            FACTOR_MAPPINGS,
            pd.DataFrame({
                'data_quality_index': [50, 75, 100],
                'model_risk_score': [
                    _calculate_expected_risk_score({'complexity_level': 'Medium', 'data_quality_index': 50, 'usage_frequency': 'Medium', 'business_impact_category': 'High'}, WEIGHTS, FACTOR_MAPPINGS),
                    _calculate_expected_risk_score({'complexity_level': 'Medium', 'data_quality_index': 75, 'usage_frequency': 'Medium', 'business_impact_category': 'High'}, WEIGHTS, FACTOR_MAPPINGS),
                    _calculate_expected_risk_score({'complexity_level': 'Medium', 'data_quality_index': 100, 'usage_frequency': 'Medium', 'business_impact_category': 'High'}, WEIGHTS, FACTOR_MAPPINGS)
                ]
            }),
            None
        ),
        # Test Case 3: Edge Case - Empty `variation_values` list
        # Expected functionality: returns an empty DataFrame with appropriate column names.
        (
            {
                'complexity_level': 'Medium',
                'data_quality_index': 80,
                'usage_frequency': 'Medium',
                'business_impact_category': 'High'
            },
            'complexity_level',
            [],  # Empty list of values to vary
            WEIGHTS,
            FACTOR_MAPPINGS,
            pd.DataFrame(columns=['complexity_level', 'model_risk_score']),
            None
        ),
        # Test Case 4: Edge Case - `param_to_vary` is not a recognized factor for scoring
        # Expected functionality: raises a KeyError if an unrecognized parameter is passed.
        (
            {
                'complexity_level': 'Medium',
                'data_quality_index': 80,
                'usage_frequency': 'Medium',
                'business_impact_category': 'High'
            },
            'non_existent_param',  # Parameter not in WEIGHTS or FACTOR_MAPPINGS
            ['Value1', 'Value2'],
            WEIGHTS,
            FACTOR_MAPPINGS,
            None,
            KeyError # Expect a KeyError when trying to access weights or factor_mappings for 'non_existent_param'
        ),
        # Test Case 5: Edge Case - Malformed `factor_mappings` (missing a required key)
        # Expected functionality: raises a KeyError if factor_mappings is missing a mapping required for scoring.
        (
            {
                'complexity_level': 'Low',
                'data_quality_index': 80,
                'usage_frequency': 'Medium',
                'business_impact_category': 'High'
            },
            'complexity_level',
            ['Low', 'High'],
            WEIGHTS,
            { # Malformed FACTOR_MAPPINGS, missing 'complexity_level' mapping
                'data_quality_index': lambda val: (100 - val) / 10,
                'usage_frequency': {'Low': 1, 'Medium': 3, 'High': 5},
                'business_impact_category': {'Low': 1, 'Medium': 3, 'High': 5, 'Critical': 10}
            },
            None,
            KeyError # Expect a KeyError when trying to access factor_mappings['complexity_level']
        ),
    ]
)
def test_perform_sensitivity_analysis(base_model_params, param_to_vary, variation_values, weights, factor_mappings, expected_output, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            perform_sensitivity_analysis(base_model_params, param_to_vary, variation_values, weights, factor_mappings)
    else:
        result_df = perform_sensitivity_analysis(base_model_params, param_to_vary, variation_values, weights, factor_mappings)
        
        # Handle potential dtype differences for empty DataFrames
        if expected_output.empty and result_df.empty:
            assert list(result_df.columns) == list(expected_output.columns)
        else:
            # Use check_exact=False and atol for floating-point comparisons
            # check_dtype=True ensures column dtypes match
            pd.testing.assert_frame_equal(result_df, expected_output, check_exact=False, atol=1e-9, check_dtype=True)


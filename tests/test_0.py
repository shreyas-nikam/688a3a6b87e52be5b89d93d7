import pytest
import pandas as pd
from definition_c25f0ca0e7664e3a9cb3c86efe0d7597 import generate_synthetic_model_data

def test_generate_synthetic_model_data_structure_and_counts():
    """
    Tests that the generated DataFrame has the correct structure, number of rows,
    column names, and unique model_ids for various valid num_models inputs,
    including edge cases like 0 and 1 model.
    """
    num_models_cases = [0, 1, 5, 100]
    expected_columns = [
        'model_id',
        'complexity_level',
        'data_quality_index',
        'usage_frequency',
        'business_impact_category'
    ]

    for num_models in num_models_cases:
        df = generate_synthetic_model_data(num_models)

        # Assert it's a Pandas DataFrame
        assert isinstance(df, pd.DataFrame)

        # Assert correct number of rows
        assert df.shape[0] == num_models

        # Assert correct column names
        assert df.columns.tolist() == expected_columns

        # Assert model_id is unique for non-empty DataFrames
        if num_models > 0:
            assert df['model_id'].is_unique, f"model_id not unique for {num_models} models"

def test_generate_synthetic_model_data_column_values():
    """
    Tests that the generated DataFrame's columns contain values within expected ranges
    or categories for a typical number of models, ensuring data integrity.
    """
    num_models = 50 # A reasonable number to ensure variety in generated values
    df = generate_synthetic_model_data(num_models)

    # Check 'complexity_level' categories
    expected_complexity_levels = ['Low', 'Medium', 'High']
    assert df['complexity_level'].isin(expected_complexity_levels).all(), \
        f"Complexity levels found: {df['complexity_level'].unique()}"

    # Check 'data_quality_index' range and type
    # Adhering to the function's own docstring (0-100)
    assert pd.api.types.is_integer_dtype(df['data_quality_index']), \
        f"data_quality_index is not an integer type: {df['data_quality_index'].dtype}"
    assert df['data_quality_index'].min() >= 0, \
        f"data_quality_index has values below 0: {df['data_quality_index'].min()}"
    assert df['data_quality_index'].max() <= 100, \
        f"data_quality_index has values above 100: {df['data_quality_index'].max()}"

    # Check 'usage_frequency' categories
    expected_usage_frequencies = ['Low', 'Medium', 'High']
    assert df['usage_frequency'].isin(expected_usage_frequencies).all(), \
        f"Usage frequencies found: {df['usage_frequency'].unique()}"

    # Check 'business_impact_category' categories
    expected_business_impact_categories = ['Low', 'Medium', 'High', 'Critical']
    assert df['business_impact_category'].isin(expected_business_impact_categories).all(), \
        f"Business impact categories found: {df['business_impact_category'].unique()}"

@pytest.mark.parametrize(
    "num_models_input, expected_exception",
    [
        ("invalid", TypeError),      # String input
        (1.5, TypeError),            # Float input
        (None, TypeError),           # None input
        ([10], TypeError),           # List input
        ({'count': 10}, TypeError)   # Dictionary input
    ]
)
def test_generate_synthetic_model_data_invalid_num_models_type(num_models_input, expected_exception):
    """
    Tests that the function raises a TypeError for invalid num_models data types.
    """
    with pytest.raises(expected_exception):
        generate_synthetic_model_data(num_models_input)

def test_generate_synthetic_model_data_negative_num_models():
    """
    Tests that the function raises a ValueError for a negative num_models input,
    as model count cannot be negative.
    """
    with pytest.raises(ValueError, match="num_models must be a non-negative integer"):
        generate_synthetic_model_data(-5)

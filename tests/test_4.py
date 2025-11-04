import pytest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from definition_d30c504e49f448c3b1c7a74582201363 import plot_risk_by_category

# Mock classes for matplotlib Axes and Figure objects
class MockAxes:
    """A minimal mock for matplotlib Axes."""
    def set_title(self, *args, **kwargs): pass
    def set_xlabel(self, *args, **kwargs): pass
    def set_ylabel(self, *args, **kwargs): pass
    def tick_params(self, *args, **kwargs): pass
    def get_figure(self): return MockFigure()
    def legend(self, *args, **kwargs): pass # Add legend method if called by seaborn
    def set(self, *args, **kwargs): pass # Add set method if called by seaborn

class MockFigure:
    """A minimal mock for matplotlib Figure."""
    def __init__(self):
        self.axes = [MockAxes()] # Mimic fig.add_subplot or fig.gca()
    def get_axes(self): return self.axes
    def suptitle(self, *args, **kwargs): pass
    def tight_layout(self, *args, **kwargs): pass
    def show(self): pass

@pytest.fixture
def sample_dataframe():
    """Provides a sample Pandas DataFrame for testing."""
    data = {
        'category_col': ['Low', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Critical'],
        'score_col': [3.5, 6.2, 4.1, 8.5, 5.9, 3.8, 9.1, 10.0],
        'other_numeric_col': [10, 20, 15, 25, 18, 12, 30, 35]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_plot_functions(mocker):
    """Mocks matplotlib and seaborn plotting functions."""
    # Mock plt.subplots to return a mock figure and axes
    mock_ax = MockAxes()
    mock_fig = MockFigure()
    mocker.patch('matplotlib.pyplot.subplots', return_value=(mock_fig, mock_ax))

    # Mock seaborn plotting functions to return the mock_ax
    mocker.patch('seaborn.barplot', return_value=mock_ax)
    mocker.patch('seaborn.heatmap', return_value=mock_ax)

    # Prevent plt.show from attempting to display a plot
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.tight_layout') # Mock tight_layout as well

    return mock_ax # Return the mock_ax for potential assertions on it

# Test 1: Valid inputs for both 'bar' and 'heatmap' plot types
@pytest.mark.parametrize("plot_type", ["bar", "heatmap"])
def test_plot_risk_by_category_success(sample_dataframe, mock_plot_functions, plot_type):
    """
    Tests that the function successfully generates a plot for valid inputs
    and returns a matplotlib Axes object, for both 'bar' and 'heatmap' types.
    """
    df = sample_dataframe
    category_col = 'category_col'
    score_col = 'score_col'

    ax = plot_risk_by_category(df, category_col, score_col, plot_type)

    assert isinstance(ax, MockAxes) # Verify an Axes object is returned
    plt.subplots.assert_called_once() # Ensure a figure/axes was created

    if plot_type == "bar":
        sns.barplot.assert_called_once()
        sns.heatmap.assert_not_called()
        # Optionally, check specific call arguments of seaborn.barplot if needed
    elif plot_type == "heatmap":
        sns.heatmap.assert_called_once()
        sns.barplot.assert_not_called()
        # Optionally, check specific call arguments of seaborn.heatmap if needed

# Test 2: Missing category_col or score_col
@pytest.mark.parametrize(
    "category_col, score_col, expected_error_msg",
    [
        ('non_existent_category', 'score_col', "Column 'non_existent_category' not found"),
        ('category_col', 'non_existent_score', "Column 'non_existent_score' not found")
    ]
)
def test_plot_risk_by_category_missing_column(
    sample_dataframe, mock_plot_functions, category_col, score_col, expected_error_msg
):
    """
    Tests that the function raises a KeyError when `category_col` or `score_col`
    do not exist in the DataFrame.
    """
    df = sample_dataframe

    with pytest.raises(KeyError, match=expected_error_msg):
        plot_risk_by_category(df, category_col, score_col, 'bar')

    # No seaborn plot function should be called if an error occurs early
    sns.barplot.assert_not_called()
    sns.heatmap.assert_not_called()
    plt.subplots.assert_not_called() # No plot setup if columns are missing

# Test 3: Invalid plot_type
def test_plot_risk_by_category_invalid_plot_type(sample_dataframe, mock_plot_functions):
    """
    Tests that the function raises a ValueError for an unsupported `plot_type`.
    """
    df = sample_dataframe
    category_col = 'category_col'
    score_col = 'score_col'
    invalid_plot_type = 'line' # Neither 'bar' nor 'heatmap'

    with pytest.raises(ValueError, match="plot_type must be 'bar' or 'heatmap'"):
        plot_risk_by_category(df, category_col, score_col, invalid_plot_type)

    sns.barplot.assert_not_called()
    sns.heatmap.assert_not_called()
    plt.subplots.assert_not_called()

# Test 4: Empty DataFrame
def test_plot_risk_by_category_empty_dataframe(mock_plot_functions):
    """
    Tests the function's behavior with an empty DataFrame.
    It should return an Axes object without error, typically showing an empty plot.
    """
    empty_df = pd.DataFrame(columns=['category_col', 'score_col'])
    category_col = 'category_col'
    score_col = 'score_col'

    ax = plot_risk_by_category(empty_df, category_col, score_col, 'bar')

    assert isinstance(ax, MockAxes)
    sns.barplot.assert_called_once() # barplot should be called even with empty data
    plt.subplots.assert_called_once()

# Test 5: Single unique value in category_col
def test_plot_risk_by_category_single_category_value(sample_dataframe, mock_plot_functions):
    """
    Tests the function's behavior when the `category_col` has only one unique value.
    It should correctly plot a single bar/heatmap entry.
    """
    df_single_category = sample_dataframe.copy()
    df_single_category['category_col'] = 'OnlyCategory' # All values are the same
    category_col = 'category_col'
    score_col = 'score_col'

    ax = plot_risk_by_category(df_single_category, category_col, score_col, 'bar')

    assert isinstance(ax, MockAxes)
    sns.barplot.assert_called_once()
    plt.subplots.assert_called_once()
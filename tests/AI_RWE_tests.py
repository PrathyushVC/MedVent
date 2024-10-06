import pytest
import pandas as pd
import numpy as np
from AI_RWE.rad_reader import scatter_with_best_fit, stat_comp, bland_alt_plot

@pytest.fixture
def sample_data():
    # Sample data for testing
    data = {
        'before': [1, 2, 3, 4, 5],
        'after': [2, 3, 4, 5, 6],
        'target': [1, 2, 3, 4, 5]
    }
    return pd.DataFrame(data)

def test_scatter_with_best_fit(sample_data):
    # Test scatter_with_best_fit function
    spearman_corr, pearson_corr = scatter_with_best_fit('before', 'after', sample_data)
    assert isinstance(spearman_corr, tuple)
    assert isinstance(pearson_corr, tuple)

def test_stat_comp_parametric(sample_data):
    # Test stat_comp function with parametric option
    results = stat_comp('before', 'after', 'target', sample_data, parametric='true')
    assert 'pearson_corr_before' in results
    assert 'pearson_corr_after' in results
    assert 'ttest' in results

def test_stat_comp_non_parametric(sample_data):
    # Test stat_comp function with non-parametric option
    results = stat_comp('before', 'after', 'target', sample_data, parametric='false')
    assert 'spearman_corr_before' in results
    assert 'spearman_corr_after' in results
    assert 'wilcoxon' in results

def test_stat_comp_all(sample_data):
    # Test stat_comp function with all option
    results = stat_comp('before', 'after', 'target', sample_data, parametric='all')
    assert 'pearson_corr_before' in results
    assert 'spearman_corr_before' in results

def test_bland_alt_plot(sample_data):
    # Test bland_alt_plot function (this will show a plot, so we just check if it runs without error)
    try:
        bland_alt_plot(sample_data['before'].to_numpy(), sample_data['after'].to_numpy())
    except Exception as e:
        pytest.fail(f"Bland-Altman plot raised an exception: {e}")

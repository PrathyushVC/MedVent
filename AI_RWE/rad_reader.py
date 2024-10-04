import pandas as pd
import numpy as np
from scipy.stats import pearsonr,spearmanr,ttest_rel,wilcoxon
import matplotlib.pyplot as plt




def scatter_with_best_fit(column1, column2, dataframe, color=None):
    """
    This function generates a scatter plot with a best fit line between two columns of a dataframe.
    
    Parameters:
    column1 (str): The name of the first column in the dataframe.
    column2 (str): The name of the second column in the dataframe.
    dataframe (pd.DataFrame): The dataframe containing the data.
    color (optional): A boolean array to color the scatter plot points differently.

    Returns:
    tuple: Spearman and Pearson correlation coefficients.
    
    The function performs the following steps:
    1. Extracts data from the dataframe.
    2. Calculates the best fit line.
    3. Calculates the R-squared value.
    4. Calculates the Spearman and Pearson correlation coefficients.
    5. Plots the scatter plot with the best fit line.
    6. Adds labels, legend, R-squared value, and correlation coefficients to the plot.
    7. Displays the plot.
    """
    
    x = dataframe[column1]
    y = dataframe[column2]

   
    m, b = np.polyfit(x, y, 1)

   
    y_pred = m * x + b
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    
    spearman_corr = spearmanr(x, y)
    pearson_corr = pearsonr(x, y)
    print(spearman_corr)
    print(pearson_corr)


  
    plt.figure(figsize=(8, 6))

   
    if color is not None:
        plt.scatter(x, y, c=np.where(color, 'red', 'blue'), alpha=0.7)
    else:
        plt.scatter(x, y, label='Data Points')

   
    plt.plot(x, m*x + b, color='black', linestyle='--', label='Best Fit Line')

 
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title(f'Scatter plot between {column1} and {column2}')
    plt.legend(loc='upper right')

    # Add R-squared value and Pearson correlation coefficient to the plot
    plt.text(0.05, 0.95, f'R-squared: {r_squared:.2f}\nPearson correlation: {pearson_corr.statistic:.2f}\nSpearman correlation: {spearman_corr.statistic:.2f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.grid(True)
    plt.show(block=False)
    #return plt
    return (spearman_corr,pearson_corr)

def stat_comp(before_col, after_col, target_col, dataframe,parametric='All'):
    """
    Performs statistical comparisons between two sets of data (before and after) 
    with respect to a target variable. The function can perform parametric and 
    non-parametric tests based on the specified parameter.

    Parameters:
    before_col (str): The name of the column representing the data before the intervention.
    after_col (str): The name of the column representing the data after the intervention.
    target_col (str): The name of the column representing the target variable for correlation.
    dataframe (pd.DataFrame): The DataFrame containing the data.
    parametric (str): Specifies the type of statistical tests to perform. 
                      Options are 'true', 'false', or 'all'. Default is 'All'.

    Returns:
    results (list): A list containing the results of the statistical tests performed.
    """

    before = dataframe[before_col].to_numpy()
    after = dataframe[after_col].to_numpy()
    target = dataframe[target_col].to_numpy()
    results={}


    if parametric.lower()=='true':
        results.append(_parametric_stats(before,after,target))

    elif parametric.lower()== 'false':
        results.append(_non_parametric_stats(before,after,target))

    elif parametric.lower()=='all':
        results.append(_parametric_stats(before,after,target))
        results.append(_non_parametric_stats(before,after,target))

    else:
        raise ValueError(f'Parametric must be one of the three options all, true, false.')
    return results

def _parametric_stats(before,after,target):
    """
    Performs parametric statistical analysis on two sets of data (before and after) 
    with respect to a target variable. This function calculates the Pearson correlation 
    coefficients for both sets of data, converts these coefficients to Fisher's Z scores, 
    and performs a paired t-test to compare the two sets.

    Parameters:
    before (np.ndarray): The array of data representing the values before the intervention.
    after (np.ndarray): The array of data representing the values after the intervention.
    target (np.ndarray): The array of data representing the target variable for correlation.

    Returns:
    dict: A dictionary containing the following keys and their corresponding values:
        - 'pearson_corr_before': Tuple containing the Pearson correlation coefficient and p-value for the before data.
        - 'pearson_corr_after': Tuple containing the Pearson correlation coefficient and p-value for the after data.
        - 'fisher_z_before': Fisher's Z score for the Pearson correlation coefficient of the before data.
        - 'fisher_z_after': Fisher's Z score for the Pearson correlation coefficient of the after data.
        - 'ttest': The t-statistic from the paired t-test comparing before and after data.
        - 'p_val_ttest': The p-value from the paired t-test.
    """
    pearson_corr_before = pearsonr(before, target)
    pearson_corr_after = pearsonr(after, target)
    fisher_z_before = _fisher_z(pearson_corr_before[0])
    fisher_z_after = _fisher_z(pearson_corr_after[0])
    ttest,p_val_ttest=ttest_rel(before,after)
    return {
        'pearson_corr_before': pearson_corr_before,
        'pearson_corr_after': pearson_corr_after,
        'fisher_z_before': fisher_z_before,
        'fisher_z_after': fisher_z_after,
        'ttest': ttest,
        'p_val_ttest': p_val_ttest
    }
def _non_parametric_stats(before,after,target):
    """
    Performs non-parametric statistical analysis on two sets of data (before and after) 
    with respect to a target variable. This function calculates the Spearman correlation 
    coefficients for both sets of data and performs the Wilcoxon signed-rank test to compare 
    the two sets.

    Parameters:
    before (np.ndarray): The array of data representing the values before the intervention.
    after (np.ndarray): The array of data representing the values after the intervention.
    target (np.ndarray): The array of data representing the target variable for correlation.

    Returns:
    dict: A dictionary containing the following keys and their corresponding values:
        - 'spearman_corr_before': Tuple containing the Spearman correlation coefficient and p-value for the before data.
        - 'spearman_corr_after': Tuple containing the Spearman correlation coefficient and p-value for the after data.
        - 'wilcoxon': The test statistic from the Wilcoxon signed-rank test comparing before and after data.
        - 'p_val_sign_rank': The p-value from the Wilcoxon signed-rank test.
    """
    spearman_corr_before = spearmanr(before, target)
    spearman_corr_after = spearmanr(after, target)
    stats_sign_rank,p_val_sign_rank=wilcoxon(before,after)

    return {
        'spearman_corr_before': spearman_corr_before,
        'spearman_corr_after': spearman_corr_after,
        'wilcoxon': stats_sign_rank,
        'p_val_sign_rank': p_val_sign_rank
    }

def bland_alt_plot(before,after,before_name='Human',after_name='Human+AI',title='Bland-Altman Plot'):
    """
    Generates a Bland-Altman plot to visualize the agreement between two sets of measurements (before and after).
    
    Parameters:
    before (np.ndarray): The array of data representing the values before the intervention.
    after (np.ndarray): The array of data representing the values after the intervention.
    before_name (str): The label for the before data in the plot.
    after_name (str): The label for the after data in the plot.
    title (str): The title of the plot.
    """
    diffs = np.array(after) - np.array(before)
    means = (np.array(after) + np.array(before)) / 2

    plt.scatter(means, diffs)
    plt.axhline(np.mean(diffs), color='gray', linestyle='--')  # Mean difference
    plt.axhline(np.mean(diffs) + 1.96*np.std(diffs), color='red', linestyle='--')  # Upper limit
    plt.axhline(np.mean(diffs) - 1.96*np.std(diffs), color='red', linestyle='--')  # Lower limit
    plt.xlabel(f'Mean of {before_name} and {after_name} Scores')
    plt.ylabel(f'Difference between {before_name} and {after_name} Scores')
    plt.title(title)
    plt.show()

def _fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))


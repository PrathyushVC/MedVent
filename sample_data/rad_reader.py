import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def scatter_with_best_fit(column1, column2, dataframe, color=None):
    # Extract data from the dataframe
    x = dataframe[column1]
    y = dataframe[column2]

    # Calculate best fit line
    m, b = np.polyfit(x, y, 1)

    # Calculate R-squared value
    y_pred = m * x + b
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate Pearson correlation coefficient
    pearson_corr = np.corrcoef(x, y)[0, 1]

    # Plot scatter plot
    plt.figure(figsize=(8, 6))

    # Scatter plot with size and color if provided
    if color is not None:
        plt.scatter(x, y, c=np.where(color, 'red', 'blue'), alpha=0.7)
    else:
        plt.scatter(x, y, label='Data Points')

    # Plot best fit line
    plt.plot(x, m*x + b, color='black', linestyle='--', label='Best Fit Line')

    # Add labels and legend
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title(f'Scatter plot between {column1} and {column2}')
    plt.legend(loc='upper right')

    # Add R-squared value and Pearson correlation coefficient to the plot
    plt.text(0.05, 0.95, f'R-squared: {r_squared:.2f}\nPearson correlation: {pearson_corr:.2f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.grid(True)
    plt.show(block=False)
    #return plt
    


# Load Excel file
excel_file = r'C:\Users\chirr\OneDrive - Case Western Reserve University\MRE_Radiology-Scoring.xlsx'

sheet1 = pd.read_excel(excel_file, sheet_name='Amit-David-Combination pruned')
sheet2 = pd.read_excel(excel_file, sheet_name='MRE-Severity-Score-Sheet')

merged_df = pd.merge(sheet1, sheet2, how='inner', left_on='patient_id', right_on='alias_mrn')
merged_df.drop(["record_id","alias_mrn"],axis=1,inplace=True)

print(merged_df)
#output_csv = 'merged_output.csv'
#merged_df.to_csv(output_csv, index=False)
pairs_fibrosis=[('fibrosis-Amit','fibrosis-David'),('fibrosis-Amit','rvas-chronic_non_inflammation_f'),(('fibrosis-David','rvas-chronic_non_inflammation_f'))]
pairs_inflammation=[('inflammation-Amit','inflammation-David'),('inflammation-Amit','rvas-inflammation'),(('inflammation-David','rvas-inflammation'))]

#results_fibrosis=pg.intraclass_corr(data=merged_df,ratings=pairs_fibrosis)
#results_inflammation=pg.intraclass_corr(data=merged_df,ratings=pairs_inflammation)

#print(results_fibrosis)
#print(results_inflammation)

scatter_with_best_fit("fibrosis-Amit", "fibrosis-David", merged_df,color=merged_df['Bin_sev_fib_co70'].values)
scatter_with_best_fit("inflamation-Amit","inflamation-David",merged_df,color=merged_df['Bin_sev_inf_co70'].values)

scatter_with_best_fit("fibrosis-Amit", "rvas-chronic_non_inflammation_f", merged_df,color=merged_df['Bin_sev_fib_co70'].values)
scatter_with_best_fit("inflamation-Amit","rvas-inflammation",merged_df,color=merged_df['Bin_sev_inf_co70'].values)

scatter_with_best_fit("fibrosis-Amit", "fibrosis-David", merged_df,color=merged_df['Bin_sev_fib_co70'].values)
scatter_with_best_fit("inflamation-Amit","inflamation-David",merged_df,color=merged_df['Bin_sev_inf_co70'].values)

scatter_with_best_fit("fibrosis-Amit", "severity_fibrosis_vas", merged_df,color=merged_df['Bin_sev_fib_co70'].values)
scatter_with_best_fit("inflamation-Amit","severity_inflammation_vas",merged_df,color=merged_df['Bin_sev_inf_co70'].values)

scatter_with_best_fit("fibrosis-David", "severity_fibrosis_vas", merged_df,color=merged_df['Bin_sev_fib_co70'].values)
scatter_with_best_fit("inflamation-David","severity_inflammation_vas",merged_df,color=merged_df['Bin_sev_inf_co70'].values)

scatter_with_best_fit("rvas-chronic_non_inflammation_f", "severity_fibrosis_vas", merged_df,color=merged_df['Bin_sev_fib_co70'].values)
scatter_with_best_fit("rvas-inflammation","severity_inflammation_vas",merged_df,color=merged_df['Bin_sev_inf_co70'].values)


#a.savefig('favd.png')
#b.savefig('iavd.png')
#c.savefig('favg.png')
#d.savefig('iavg.png')
#e.savefig('fdvg.png')
#f.savefig('idvg.png')




input("Press Enter to continue...")

# ANOVA and Tukey Range Tests
# Allows us to investigate an association between a quantitative variable(order amount) and a non-binary categorical variable (stores)

# Assumptions of T-test, ANOVA and Tukey
# 1. The observations should be independently randomly sampled from the population
# 2. The standard deviations of the groups should be equal
# 3. The data should be normally distributed
# 4. The groups created by the categorical variable must be independent

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import amazing_functions as af
from matplotlib.backends.backend_pdf import PdfPages
import subprocess
import os
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Install statsmodels
# Used the subprocess module to run the pip install command to instal the statsmodels package.Commenting out since I installed the package
#subprocess.check_call(['pip', 'install', 'statsmodels'])

# Now you can import and use statsmodels in your script
#import statsmodels.api as sm

# Set a random seed for reproducibility
np.random.seed(987123)

# Generate a fake dataset for 10 stores with different order amounts and 200 orders per store
num_stores = 10
data = pd.DataFrame({
    'Store': np.repeat(range(1, num_stores + 1), 200),  # 200 orders per store
    'OrderAmount': np.random.normal(loc=50, scale=20, size=num_stores * 200)
})

# Make two stores (e.g., Store 5 and Store 8) have a different distribution
# If this code isn't added, we would probably fail to reject the null hypothesis
data.loc[data['Store'] == 5, 'OrderAmount'] = np.random.normal(loc=70, scale=25, size=200)
data.loc[data['Store'] == 8, 'OrderAmount'] = np.random.normal(loc=30, scale=15, size=200)

# Create boxplot for initial visualization
plt.figure(figsize=(12, 6))
ax = sns.boxplot(x='Store', y='OrderAmount', data=data)
sns.boxplot(x='Store', y='OrderAmount', data=data)
plt.title('Boxplot of Order Amounts for 10 Stores')
plt.xlabel('Store')
plt.ylabel('Order Amount')

# Save the plot to a file in a different directory using amazing_functions
# Add annotations to highlight means and distributions
for store_num in [5, 8]:
    store_median = data[data['Store'] == store_num]['OrderAmount'].median()
    ax.annotate(f'Median: {store_median:.2f}', xy=(store_num-1, store_median), xytext=(store_num - 0.7, store_median + 27),
                arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9, color='red')

# Add text annotation to explain why we reject the null hypothesis
ax.text(0.7, 0.92, 'Two stores with significantly different\norder amount distributions. These result in a small\np-value and rejection of the null hypothesis of the ANOVA test', 
        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='blue')

af.back_one_enter_new('Python_Generated_Images', 'boxplot_Visualizing_ANOVA_result.png')
plt.show()

current_directory = os.path.dirname(os.path.abspath(__file__))
output_directory = os.path.join(current_directory, '..', 'Python_Generated_Images')
output_filename = 'anova_and_tukey_results.pdf'
output_filepath = os.path.join(output_directory, output_filename)
with PdfPages(output_filepath) as pdf:
    # Create boxplot for initial visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Store', y='OrderAmount', data=data, ax=ax)
    plt.title('Boxplot of OrderAmount for 10 Stores')
    plt.xlabel('Store')
    plt.ylabel('OrderAmount')
    
    # Save the boxplot
    pdf.savefig()
    plt.close(fig)

# Perform one-way ANOVA
# The f_oneway function takes multiple groups (in this case, the order amounts for each store) and tests whether the means of these groups are significantly different.
# This is a list comprehension that creates a list of order amounts for each store
# The * operator is used for unpacking the list, passing each list of order amounts as a separate argument to the f_oneway function.
    anova_result = scipy.stats.f_oneway(*[data['OrderAmount'][data['Store'] == store] for store in range(1, num_stores + 1)])

# Display ANOVA results
    print(f'One-way ANOVA results:')
    print(f'F-statistic: {anova_result.statistic:.4f}')
    print(f'p-value: {anova_result.pvalue:.4f}')

    tukey_results = pairwise_tukeyhsd(data.OrderAmount, data.Store, 0.05)
    # Save the Tukey test results
    stores = data['Store'].unique()
    for i in stores:
        fig, ax = plt.subplots(figsize=(10, 6))
        tukey_results.plot_simultaneous(comparison_name=i, xlabel='Difference in Means', ax=ax)
        plt.title(f'Tukey HSD Test Results in relation to Store {i}')
        pdf.savefig()
        plt.close(fig)


# Interpretation of ANOVA results
alpha = 0.05
if anova_result.pvalue < alpha:
    print('The p-value is less than the significance level.')
    print('There is a significant difference in mean order amounts between at least two stores.')
else:
    print('The p-value is greater than or equal to the significance level.')
    print('We fail to reject the null hypothesis. There is no significant difference in mean order amounts between stores.')

# At least one pair of stores has a significant difference in mean order amounts
# The next step is to find out which pair of stores differ.
# We can perform Tukey's range test to do this.
# Tukey’s range test is similar to running 50 separate 2-sample t-tests, except that it runs all of these tests simultaneously in order to preserve the type I error rate.
# The function output is a table, with one row per pair-wise comparison\
# There is a significant difference between every store and store 5 and every store vs store 8

print(tukey_results.summary())







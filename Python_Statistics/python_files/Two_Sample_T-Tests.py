# Two Sample T-Tests
# Association between a quantitative variable and a binary categorical variable
# Used to determine if there is a significant difference between the means of two independent groups.
# Assess whether the observed differences are likely due to chance or if they represent a true difference in the populations from which the samples were drawn

import numpy as np
import pandas as pd
import amazing_functions as af

# I currently work for Amazon Fresh - Amazon's grocery story

# Let's create a simplified fake dataset for orders from two different versions of the Amazon Fresh website.
# We'll assume the key metric of interest is the order amount. For learning purposes, we'll make the distributions
# of order amounts different for the two versions.

# Steps
# 1. Formulate Hypotheses:
# Null Hypothesis (H0): There is no significant difference between the mean order amounts of the two groups.
# Alternative Hypothesis (H1): There is a significant difference between the means order amounts of the two groups.

# 2.Collect Data:
# Obtain a sample from each of the two groups under consideration. These groups should be independent of each other.
# Set a random seed for reproducibility
np.random.seed(2300)

# Assume these are random samples of all Amazon Fresh customers
# Generate fake data for the old version
old_version_orders = np.random.normal(loc=50, scale=20, size=200)
# Generate fake data for the new version with a higher mean
new_version_orders = np.random.normal(loc=70, scale=25, size=200)

# Combine the data into a DataFrame
data = pd.DataFrame({
    'Version': ['Old'] * 200 + ['New'] * 200,
    'OrderAmount': np.concatenate([old_version_orders, new_version_orders])
})

# Display the first few rows of the dataset
# print(data.head())

# Group by 'Version' and calculate the mean order amount for each version
means_by_version = data.groupby('Version')['OrderAmount'].mean()

# Display the mean order amounts for each version
print(means_by_version)

# 3. Calculate the T-statistic:
# Use the formula for the t-statistic, which involves the difference between the sample means divided by the standard error of the difference.
# Or, We can use the scipy.stats module in Python to perform a two-sample t-test with the generated dataset.

import scipy.stats

# Separate the data into two groups
# Use boolean indexing to create two separate arrays of order amounts for the 'Old' and 'New' versions.
old_version_data = data[data['Version'] == 'Old']['OrderAmount']
new_version_data = data[data['Version'] == 'New']['OrderAmount']

# Perform a two-sample t-test
# Use the ttest_ind function from scipy.stats to perform the two-sample t-test
t_statistic, p_value = scipy.stats.ttest_ind(old_version_data, new_version_data)

# The t-statistic indicates the size of the difference in order amount relative to the variability in the data.
# The p-value assesses the evidence against a null hypothesis. A small p-value (typically less than 0.05) suggests that you can reject the null hypothesis in favor of the alternative hypothesis.
# Display the results
print(f'T-statistic: {t_statistic}')
print(f'P-value: {p_value}')

# In this case, the p-value is much smaller than 0.05, so we confidently reject the null hypothesis that the order amounts are the same for the two versions of the website.
# There is a signficant difference between the mean order amounts of the two groups

# Set the significance level
alpha = 0.05

# Check the p-value and draw a conclusion
if p_value < alpha:
    print(f'The p-value ({p_value:.4f}) is less than the significance level ({alpha}), so we reject the null hypothesis.')
    print('There is a statistically significant difference in mean order amounts between the old and new versions.')
else:
    print(f'The p-value ({p_value:.4f}) is greater than or equal to the significance level ({alpha}).')
    print('We fail to reject the null hypothesis. There is no statistically significant difference in mean order amounts.')


import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for better visualization
sns.set(style="whitegrid")

# Create overlapping histograms using seaborn
plt.figure(figsize=(10, 6))
ax = sns.histplot(data, x='OrderAmount', hue='Version', element="step", stat="density", common_norm=False, kde=True)

# Add text annotations for means and p-value
old_mean = old_version_data.mean()
new_mean = new_version_data.mean()
p_value_annotation = f'p-value: {p_value:.4f}'

ax.text(old_mean, 0.02, f'Old Mean: {old_mean:.2f}', verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=10)
ax.text(new_mean, 0.02, f'New Mean: {new_mean:.2f}', verticalalignment='bottom', horizontalalignment='left', color='orange', fontsize=10)
ax.text(0.5, 0.1, p_value_annotation, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=12)

# Add a context explanation for the p-value
ax.text(0.5, 0.05, 'A small p-value suggests a significant difference\nin mean order amounts between versions.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='black', fontsize=10)

# Add labels and a title
plt.xlabel('Order Amount')
plt.ylabel('Density')
plt.title('Overlapping Histograms of Order Amounts for Old and New Versions of Amazon Fresh Website (Fake Data)')

# Save the plot to a file in a different directory using amazing_functions
af.back_one_enter_new('Python_Generated_Images', 'overlapping_histograms_difference_btw_groups.png')

# Show the plot
plt.show()

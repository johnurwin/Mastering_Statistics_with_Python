# AB Tests
# An A/B Test is a method used to compare two versions of something.
# The goal is to determine which version performs better
# Example: What number of products on a website makes customers most likely to purchase something: 50 or 100?
# What color button are customers more likely to click on: blue or green?
# Do people spend more time on a language learning app if the main character is a penguin or a monkey?

# For A/B tests where the outcome of interest is categorical, we use a Chi-Square hypothesis test.
# This is because we are comparing a categorical variables to a categorical variable

# 1.) Step one of this kind of test is to determine the sample size needed to perform the test
# Need these three numbers:
# Baseline conversion rate - E.g. The percent of people who make a purchase under the current website design.
# minimum detectable effect - E.g. The smallest difference that we care to measure (desired lift) ... Could also be compared to our baseline conversion rate
# statistical significance threshold - E.g. 0.05 is a common threshold (False positive rate)
# Keep in mind that there is a trade off between false positive rate and false negative rate
# As significance threshold (False Positive Rate) decreases, a greater sample size is needed
# As baseline conversion rate decreases, a greater sample size is needed
# And as minimum detectable effect decreases, a greater sample size is needed


# Suppose that you are running a business and want to see if you have a hire conversion rate when 50 products are on the webite or if 100 products are on the website.
# You are testing two versions of the website and trying to determine if doubling the products increases conversion rate

# To do this, generate a fake dataset for an A/B test scenario with two versions of a website (one with 50 products and another with 100 products),
# you can use the numpy and pandas libraries in Python.

import numpy as np
import pandas as pd
import amazing_functions as af

# Set a random seed for reproducibility
np.random.seed(42)

# Number of visitors in each group
num_visitors_per_group = 1000

# Generate data for the website with 50 products
website_50_products = pd.DataFrame({
    'VisitorID': np.arange(1, num_visitors_per_group + 1),
    'Converted_Order': np.random.choice([0, 1], num_visitors_per_group, p=[0.8, 0.2])  # 20% conversion rate
})

# Generate data for the website with 100 products
website_100_products = pd.DataFrame({
    'VisitorID': np.arange(1, num_visitors_per_group + 1),
    'Converted_Order': np.random.choice([0, 1], num_visitors_per_group, p=[0.8, 0.2])  # 20% conversion rate
})

# Display sample data
print("Website with 50 products:")
print(website_50_products.head())

print("\nWebsite with 100 products:")
print(website_100_products.head())

# Concatenate the two datasets and add a 'WebsiteVersion' column
website_50_products['WebsiteVersion'] = '50_products'
website_100_products['WebsiteVersion'] = '100_products'

# Concatenate the datasets
combined_dataset = pd.concat([website_50_products, website_100_products], ignore_index=True)

# Display the combined dataset
print("Combined Dataset:")
print(combined_dataset.head())

# Compare the conversion rates between the website with 50 products and the website with 100 products
# The null hypothesis is that there is no significant difference in conversion rates between the two versions. 
# If the p-value is less than the chosen significance level (alpha), you may reject the null hypothesis and conclude that there is a statistically significant difference.

from scipy.stats import chi2_contingency
import scipy.stats

# Create a contingency table for the A/B test
contingency_table = pd.crosstab(combined_dataset['WebsiteVersion'], combined_dataset['Converted_Order'])

# Display the contingency table
print("Contingency Table:")
print(contingency_table)

# Perform chi-squared test
chi2, p, _, _ = chi2_contingency(contingency_table)

# Display the test statistics and p-value
print("\nChi-squared test statistics:", chi2)
print("P-value:", p)

# Determine statistical significance
alpha = 0.05
if p < alpha:
    print("\nThere is a statistically significant difference between the two website versions.")
else:
    print("\nThe difference between the two website versions is not statistically significant.")



# We forgot to calculate the baseline conversion rate, minimum detectable effect, statistical significance threshold, and the sample size needed for the A/B test,
# you'll need to define these parameters based on your specific goals and requirements.

from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower

# Define parameters
baseline_conversion_rate = 0.2  # 20% conversion rate
minimum_detectable_effect = 0.02  # 2% absolute improvement
statistical_significance_threshold = 0.05  # 5%
power = 0.8  # desired statistical power

# Calculate the sample size needed for each group using power analysis
effect_size = minimum_detectable_effect / baseline_conversion_rate
alpha = statistical_significance_threshold
nobs1 = num_visitors_per_group

normal_power = NormalIndPower()
required_sample_size_per_group = normal_power.solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    ratio=1,  # Assuming equal group sizes
    alternative='larger'  # Assuming a one-tailed test for improvement
)

# Display results
print(f"Baseline Conversion Rate: {baseline_conversion_rate * 100:.2f}%")
print(f"Minimum Detectable Effect: {minimum_detectable_effect * 100:.2f}%")
print(f"Statistical Significance Threshold: {statistical_significance_threshold * 100:.2f}%")
print(f"Power: {power * 100:.2f}%")
print(f"Required Sample Size per Group: {required_sample_size_per_group:.0f}")


# Let's create a hypothetical situation where we want to test the impact of changing the the call-to-action (CTA) button from a penguin character to a different animal, let's say a panda.
# We'll assume that the penguin button has a click-through rate of 10%, and we want to test if changing it to a panda button has any significant impact.


# Number of visitors in each group
num_visitors_per_group = 1237

# Generate data for the original website with a penguin button
original_website = pd.DataFrame({
    'VisitorID': np.arange(1, num_visitors_per_group + 1),
    'Clicked': np.random.choice([0, 1], num_visitors_per_group, p=[0.9, 0.1])  # 10% click-through rate
})

# Generate data for the modified website with a panda button
modified_website = pd.DataFrame({
    'VisitorID': np.arange(1, num_visitors_per_group + 1),
    'Clicked': np.random.choice([0, 1], num_visitors_per_group, p=[0.92, 0.08])  # 8% improvement in click-through rate
})

# Concatenate the datasets and add an 'AnimalButton' column
original_website['AnimalButton'] = 'Penguin'
modified_website['AnimalButton'] = 'Panda'
combined_data = pd.concat([original_website, modified_website], ignore_index=True)

# Display sample data
print("Combined Dataset:")
print(combined_data.head())

# Create a contingency table for the A/B test
contingency_table = pd.crosstab(combined_data['AnimalButton'], combined_data['Clicked'])

# Display the contingency table
print("\nContingency Table:")
print(contingency_table)

# Perform chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Display the test statistics and p-value
print("\nChi-squared test statistics:", chi2)
print("P-value:", p)
print("dof:", dof)
print("expected:", expected)

# The A/B test uses a chi-squared test to compare the click-through rates for the two animal buttons.
# Determine statistical significance
alpha = 0.05
if p < alpha:
    print("\nThere is a statistically significant difference between the animal buttons. Use the new panda button")
else:
    print("\nThe difference between the animal buttons is not statistically significant. Continue to use the current animal button")

import matplotlib.pyplot as plt
import seaborn as sns

# Stacked bar plot for clicks and non-clicks
contingency_table.plot(kind='bar', stacked=True, color=['#3498db', '#e74c3c'])
plt.title('Clicks and Non-Clicks by Animal Button')
plt.xlabel('Animal Button')
plt.ylabel('Count')
plt.legend(title='Clicked', loc='upper right')

# Save the plot to a file in a different directory using amazing_functions
af.back_one_enter_new('Python_Generated_Images', 'bar_plot_AB_Test.png')

plt.show()


# Heatmap for the contingency table
# In order to run a hypothesis test to decide whether there is a significant difference in the click rate between these animal buttons, we would run a Chi-Square test.
# To accomplish this, we would first create a contingency table for the AnimalButton and Clicked variables in the above table:
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, cmap="Blues", fmt='g', cbar=False)
plt.title('Contingency Table for A/B Test')
plt.xlabel('Clicked')
plt.ylabel('Animal Button')

# Save the plot to a file in a different directory using amazing_functions
af.back_one_enter_new('Python_Generated_Images', 'contingency_table_AB_Test.png')

plt.show()

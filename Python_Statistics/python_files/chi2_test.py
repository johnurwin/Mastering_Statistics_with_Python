# chi-square test for independence. This test is appropriate when you have categorical data and you want to
# assess whether there is a significant association between two categorical variables.
# Suppose we want to know whether Competitor Presence is associated with the probability of a store visitor making a purchase.
# The null hypothesis is that there’s no association between the variables.

# Null Hypothesis (H0): There is no association between Competitor Presence and the probability of making a purchase.
# Alternative Hypothesis (H1): There is an association between Competitor Presence and the probability of making a purchase.

# Is there an association between competitors in the area and purchases?

# 1. The observations should be independently randomly sampled from the population
# 2. The categories of both variables must be mutually exclusive
# 3. The groups should be independent

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Set a random seed for reproducibility
np.random.seed(51)

# Generate a fake dataset
num_visitors = 200
competitor_presence = np.random.choice([0, 1], size=num_visitors, p=[0.4, 0.6])  # 0: Competitor absent, 1: Competitor present
purchase = np.random.choice([0, 1], size=num_visitors, p=[0.3, 0.7])  # 0: No purchase, 1: Purchase

# Create a DataFrame
df = pd.DataFrame({'Competitor_Presence': competitor_presence, 'Purchase': purchase})

# Display the first few rows of the dataset
print(df.head())

# Create a contingency table
contingency_table = pd.crosstab(df['Competitor_Presence'], df['Purchase'])

# Display the contingency table
print("Contingency Table:")
print(contingency_table)

# Perform Chi-Square Test
chi2, p, _, _ = chi2_contingency(contingency_table)

# Display test results
print("\nChi-square Test Results:")
print(f'Chi-square statistic: {chi2:.4f}')
print(f'p-value: {p:.4f}')

# Interpret the results
alpha = 0.05
if p < alpha:
    print(f"\nSince p-value ({p:.4f}) is less than alpha ({alpha}), reject the null hypothesis.")
    print("There is a significant association between Competitor Presence and the probability of making a purchase.")
else:
    print(f"\nSince p-value ({p:.4f}) is greater than or equal to alpha ({alpha}), fail to reject the null hypothesis.")
    print("There is no significant association between Competitor Presence and the probability of making a purchase.")



# In Reality, I don't think we would have data for when a customer does not make a purchase for the Instore business. 
# Lets do the same test for competitor presence and the order amount

# Generate a fake dataset
num_visitors = 200
competitor_presence = np.random.choice([0, 1], size=num_visitors, p=[0.4, 0.6])  # 0: Competitor absent, 1: Competitor present
order_amount = np.random.normal(loc=50, scale=20, size=num_visitors)  # Simulating order amounts

# Ensure order amounts are non-negative
order_amount = np.maximum(order_amount, 0)

# Create a DataFrame
df = pd.DataFrame({'Competitor_Presence': competitor_presence, 'OrderAmount': order_amount})

# Display the first few rows of the dataset
print(df.head())

# Create a contingency table (for illustrative purposes, though not strictly applicable)
# chi-square tests are typically used for categorical variables, and the continuous nature of order amounts might not be fully captured by this test
# A two-sample t-test would be better because this test is for a binary categorical variable and a quantitative variable
contingency_table = pd.crosstab(df['Competitor_Presence'], df['OrderAmount'] > 50)

# Display the contingency table
print("Contingency Table:")
print(contingency_table)

# Perform Chi-Square Test (for illustrative purposes, though not typically used for continuous variables)
chi2, p, _, _ = chi2_contingency(contingency_table)

# Display test results
print("\nChi-square Test Results:")
print(f'Chi-square statistic: {chi2:.4f}')
print(f'p-value: {p:.4f}')

# Interpret the results
alpha = 0.05
if p < alpha:
    print(f"\nSince p-value ({p:.4f}) is less than alpha ({alpha}), reject the null hypothesis.")
    print("There is a significant association between Competitor Presence and the probability of making a purchase.")
else:
    print(f"\nSince p-value ({p:.4f}) is greater than or equal to alpha ({alpha}), fail to reject the null hypothesis.")
    print("There is no significant association between Competitor Presence and the probability of making a purchase.")

# Lets calculate statistics of central tendency in python
# First, lets randomly generate a dataframe of our own data
# This code generates random samples for each distribution and stores them in a DataFrame called df.
# The distributions are then visualized using histograms.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import amazing_functions as af

# Set a seed for reproducibility
np.random.seed(3800)

# Number of samples for each distribution
num_samples = 1000

# Generate random samples for each distribution

normal_samples = np.random.normal(loc=0, scale=1, size=num_samples) # loc: mean of the distribution | scale: standard deviation of the distribution
uniform_samples = np.random.uniform(low=0, high=1, size=num_samples) # low: lower bound of the distribution | high: upper bound of the distribution
binomial_samples = np.random.binomial(n=10, p=0.5, size=num_samples)# n: number of trials | # p: probability of success on each trial
poisson_samples = np.random.poisson(lam=5, size=num_samples) # lam: average rate of events occurring
exponential_samples = np.random.exponential(scale=2, size=num_samples) # scale: scale parameter, which is the inverse of the rate parameter
gamma_samples = np.random.gamma(shape=2, scale=2, size=num_samples)# shape: shape parameter | scale: scale parameter
lognormal_samples = np.random.lognormal(mean=0, sigma=1, size=num_samples) # mean: mean of the natural logarithm of the distribution | sigma: standard deviation of the natural logarithm of the distribution
# The Beta distribution is defined on the interval [0, 1], making it suitable for modeling random variables representing proportions or probabilities.
# The a parameter (alpha) and b parameter (beta) control the shape of the distribution. These parameters shape the probability density function,
# with higher values of a and b leading to a more peaked distribution around its mean.
beta_samples = np.random.beta(a=2, b=5, size=num_samples) # a: alpha parameter | b: beta parameter
# The Cauchy distribution is a symmetric distribution with heavy tails and undefined mean and variance.
# The shape of the distribution is entirely determined by its location parameter, which is 0 for standard_cauchy in NumPy. 
cauchy_samples = np.random.standard_cauchy(size=num_samples)
chi_square_samples = np.random.chisquare(df=3, size=num_samples) # df: degrees of freedom


# Create a DataFrame
data = {
    'Normal': normal_samples,
    'Uniform': uniform_samples,
    'Binomial': binomial_samples,
    'Poisson': poisson_samples,
    'Exponential': exponential_samples,
    'Gamma': gamma_samples,
    'LogNormal': lognormal_samples,
    'Beta': beta_samples,
    'Cauchy': cauchy_samples,
    'ChiSquare': chi_square_samples,
    'LowMeanHighMedian': np.concatenate([np.random.normal(loc=15, scale=1, size=num_samples//2),
                                         np.random.normal(loc=5, scale=5, size=num_samples//2)]),  # Distribution with a high mean and low median
    'HighMeanLowMedian': np.concatenate([np.random.normal(loc=-5, scale=1, size=num_samples//2),
                                         np.random.normal(loc=5, scale=5, size=num_samples//2)]),  # Distribution with a low mean and high median
}

df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(df.head())

# Calculate the mean of each column in the DataFrame 
column_means = df.mean()

# Display the mean values
print("Mean of Each Column:")
print(column_means)

# Optionally, you can visualize the distributions using histograms
# This code is ran in a .py file, so you will need to click "x" to escape from the graph before the rest of the code will execute
df.hist(bins=20, figsize=(12, 10))
plt.suptitle('Random Samples from Different Distributions')
plt.show()


# You can also use numpy to calculate the means of each column
# First, you need to convert the DataFrame to a NumPy array
data_array = df.to_numpy()

# Calculate the mean along axis 0 (columns)
column_means_np = np.mean(data_array, axis=0)

# Display the mean values
print("Mean of Each Column (NumPy):")
print(column_means_np)

# You can do the same for the median
# Calculate the mean along axis 0 (columns)
column_medians_np = np.median(data_array, axis=0)

# Display the median values
print("Median of Each Column (NumPy):")
print(column_medians_np)

# Mode
# You can use the mode() method provided by pandas
# Calculate the mode of each column in the DataFrame
# Commenting out the mode
# If there are two modes, it will record them both, so .iloc is needed
# column_modes = df.mode().iloc[0]

# Display the mode values
# print("Mode of Each Column (Pandas):")
# print(column_modes)

# You can also calculate the mode using SCIPY
from scipy import stats

# Calculate the mode of each column in the DataFrame
column_modes_scipy, column_mode_counts = stats.mode(df, axis=0, nan_policy='omit')

# Display the mode values
print("Mode and Count of Each Column (Scipy):")
print(column_modes_scipy)
print(column_mode_counts)

# Create histograms with mean and median lines
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
fig.suptitle('Histograms with Mean and Median Lines')

# histograms for each column are created, and dashed lines are added to indicate the mean (in red) and median (in green) values.
# iterates over each column of the DataFrame and create a subplot for each one in a 3x4 grid
# enumerate is used to get both the index i and the corresponding tuple (col, ax) during iteration
# df.columns provides an iterable of column names in the DataFrame.
# axes.flatten() flattens the 2D array of subplots into a 1D array for easier iteration.

for i, (col, ax) in enumerate(zip(df.columns, axes.flatten())):
    # Plot histogram
    ax.hist(df[col], bins=20, density=True, alpha=0.7, color='blue')

    # Plot mean and median lines
    mean_value = df[col].mean()
    median_value = df[col].median()
    ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
    ax.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')

    # Add labels and legend
    ax.set_title(col)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the plot to a file in a different directory using amazing_functions
af.back_one_enter_new('Python_Generated_Images', 'all_histograms_with_mean_medians.png')

# Show the plot
plt.show()

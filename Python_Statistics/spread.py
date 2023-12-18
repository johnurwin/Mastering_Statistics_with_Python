import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set a seed for reproducibility
np.random.seed(42)

# Number of samples for each distribution
num_samples = 1000

# In this script, we will apply measures of spread to each column
# We might as well learn about some interesting distributions of data as well
# Generate random samples for each distribution

# Weibull Distribution
# Parameters: shape (c) and scale (λ)
# Shape Parameter (a): Influences the shape of the distribution.
# Often used in reliability engineering to model time until failure of components, such as the lifespan of electronic devices. (life data analysis, survival analysis)
weibull_samples = np.random.weibull(a=2, size=num_samples)

# Triangular Distribution
# Parameters (left, mode, right): Defines the range and mode of the distribution.
# Useful in risk analysis and estimation when the true distribution is unknown but limited information is available.
triangular_samples = np.random.triangular(left=0, mode=0.5, right=1, size=num_samples)

# Logistic Distribution
# Parameters (loc, scale): Affects the location and scale of the distribution.
# Commonly used in logistic regression models and probability density estimation.
logistic_samples = np.random.logistic(loc=0, scale=1, size=num_samples)

# Hypergeometric Distribution
#Used in quality control, genetics, and finite population sampling, where objects are drawn from a finite population without replacement.
hypergeometric_samples = np.random.hypergeometric(ngood=30, nbad=70, nsample=10, size=num_samples)

# Multinomial Distribution
# Applied in experiments with multiple categorical outcomes
multinomial_samples = np.random.multinomial(n=20, pvals=[0.2, 0.3, 0.5], size=num_samples)

# Von Mises Distribution
# Parameters (mu, kappa): Defines the mean direction and concentration.
# Commonly used in circular statistics, representing angles or directions.
vonmises_samples = np.random.vonmises(mu=0, kappa=4, size=num_samples)

# Wald (Inverse Gaussian) Distribution
# Parameters (mean, scale): Affects the mean and scale of the distribution.
# Used in modeling financial data, such as stock prices, and in the analysis of response times.
wald_samples = np.random.wald(mean=1, scale=2, size=num_samples)

# Pareto Distribution
# Parameter (a): Determines the shape of the distribution.
# Commonly used to model distributions where a small number of items have a large impact, such as income distribution or wealth distribution.
pareto_samples = np.random.pareto(a=2, size=num_samples)

# Laplace Distribution
# Parameters (loc, scale): Influences the location and scale of the distribution.
# Used in signal processing, Bayesian statistics, and image compression.
laplace_samples = np.random.laplace(loc=0, scale=1, size=num_samples)

# Gumbel Distribution
# Applied in extreme value theory, modeling extreme values in various fields, such as environmental sciences and finance.
# Extreme value theory deals with the statistical modeling of extreme events, such as the maximum flood level in a river, the maximum wind speed in a storm,
# or the maximum value in a financial time series.
# EVT provides tools and models for analyzing and predicting the probability of such extreme events.
# In practical terms, fitting observed data to the Gumbel distribution allows statisticians and researchers to make predictions about the likelihood of extreme
# events occurring in the future based on historical data. It is a valuable tool in risk assessment and management in various fields.
gumbel_samples = np.random.gumbel(loc=0, scale=1, size=num_samples)

# Create a DataFrame
interesting_distributions = {
    'Weibull': weibull_samples,
    'Triangular': triangular_samples,
    'Logistic': logistic_samples,
    'Hypergeometric': hypergeometric_samples,
    'Multinomial': multinomial_samples[:, 0],  # Take only one column for simplicity
    'VonMises': vonmises_samples,
    'Wald': wald_samples,
    'Pareto': pareto_samples,
    'Laplace': laplace_samples,
    'Gumbel': gumbel_samples,
}

distributions_df = pd.DataFrame(interesting_distributions)

# Measures of spread
# The range is a measure of spread, but it only considers two data points in the calculation: the maximum and minimum
# We want to understand spread with one number that considers all points in the dataset

# Calculate the variance of each column in the DataFrame
# We can do this using numpy or pandas. Lets use pandas' .var() method
column_variances = distributions_df.var()

# Display the variance values
print("Variance of Each Column:")
print(column_variances)

# Calculate the standard deviation of each column in the DataFrame
column_std_devs = distributions_df.std()

# Display the standard deviation values
print("Standard Deviation of Each Column:")
print(column_std_devs)


# Create histograms with mean, one std above, and one std below lines
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
fig.suptitle('Histograms with Mean and One Standard Deviation Lines')

for i, (col, ax) in enumerate(zip(distributions_df.columns, axes.flatten())):
    # Plot histogram
    ax.hist(distributions_df[col], bins=20, density=True, alpha=0.7, color='blue')

    # Plot mean line
    mean_value = distributions_df[col].mean()
    ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')

    # Plot one std above and below lines
    std_dev_value = distributions_df[col].std()
    ax.axvline(mean_value + std_dev_value, color='green', linestyle='dashed', linewidth=2, label=f'+1 Std Dev: {mean_value + std_dev_value:.2f}')
    ax.axvline(mean_value - std_dev_value, color='green', linestyle='dashed', linewidth=2, label=f'-1 Std Dev: {mean_value - std_dev_value:.2f}')
    ax.axvline(mean_value + std_dev_value*2, color='purple', linestyle='dashed', linewidth=2, label=f'+2 Std Dev: {mean_value + std_dev_value*2:.2f}')
    ax.axvline(mean_value - std_dev_value*2, color='purple', linestyle='dashed', linewidth=2, label=f'-2 Std Dev: {mean_value - std_dev_value*2:.2f}')

    # Add labels and legend
    ax.set_title(col)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the plot as an image
plt.savefig('histograms_with_lines.png')

# Show the plot
plt.show()


# Create boxplots for all columns
distributions_df.boxplot(figsize=(15, 10), vert=False)
plt.title('Boxplots for All Columns')
plt.xlabel('Value')
plt.savefig('Boxplots_of_distributions.png')
plt.show()

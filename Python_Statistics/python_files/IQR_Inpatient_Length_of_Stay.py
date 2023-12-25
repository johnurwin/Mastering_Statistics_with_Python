# From my experience working at the Cleveland Clinic, I learned about different distributions of data
# One distribution that I found interesting was the distribution of length of stay. There were some extremely large outliers in the data
# Lets create a distribution that simulates inpatient length of stay for all hospital stays in a given time period.
# If we worked at a hospital, we may pull all inpatient stays for the full year, then analyze length of stay by certain procedures or groups of procedures
# To simulate a Length of Stay distribution, we need a distribution that will generate right-skewed data with some large outliers on the high end
# The gamma distribution is used to generate the length of stay data, which is right-skewed.
# We will plot the LOS distribution before and after dropping values outside of the Interquartile-range (IQR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import os
import amazing_functions as af


# Set seed for reproducibility
np.random.seed(333)

# Generate a right-skewed dataset with outliers
# Number of beds at the Cleveland Clinic according to their website
# Let generate the same numbers of samples as the number of beds at the hospital system
total_beds = 6749
# According to the state of the clinic in 2020, Average acute length of stay at the Cleveland Clinic was 4.92
# We will use this value in the dataset generation step
average_length_of_stay = 4.92

# The shape parameter ('a') determines the shape of the gamma distribution.
# If a is small, the distribution is skewed to the right.
# If a is large, the distribution is skewed to the left.
shape = 1.5
# The scale parameter ('β') determines the spread or scale of the distribution.
# Larger values of β result in a more spread-out distribution.
# Smaller values of β result in a more concentrated distribution.
scale = 5  
outliers_probability = 0.03  # Probability of having outliers

# Calculate shape (a) and scale (beta) based on the mean
beta = average_length_of_stay / shape
# Generate length of stay data using the gamma distribution
# Instead of scale, use beta
length_of_stay = np.random.gamma(shape, beta, total_beds)
# length_of_stay = np.random.gamma(shape, scale, num_samples)+1

# Outliers are introduced with a small probability by selecting random values from the generated dataset and multiplying them by a factor (outliers_factor) to make them larger.
outliers = np.random.choice(length_of_stay, size=int(outliers_probability * total_beds), replace=False)
outliers_factor = 7  # Factor by which outliers are larger
outliers_indices = np.random.choice(total_beds, size=int(outliers_probability * total_beds), replace=False)
length_of_stay[outliers_indices] *= 5

# Calculate the first and third quartiles
q1 = np.percentile(length_of_stay, 25)
q3 = np.percentile(length_of_stay, 75)

# Calculate the interquartile range (IQR)
iqr = q3 - q1

# Filter data based on the IQR
filtered_data = length_of_stay[(length_of_stay >= q1 - 1.5 * iqr) & (length_of_stay <= q3 + 1.5 * iqr)]

# Set the size and DPI of the figure
fig = plt.figure(figsize=(16, 10), dpi=100)  # Adjust the size and DPI as needed

# Create a histogram to visualize the original distribution
plt.subplot(3, 1, 1)
plt.hist(length_of_stay, bins=30, color='blue', edgecolor='black', density=True)
plt.xlabel('Length of Stay (Days)')
plt.ylabel('Probability Density')
plt.title(f'Original Simulated Gamma Distribution of Length of Stay at Cleveland Clinic ({total_beds} Beds) with Outliers')

plt.subplot(3, 1, 2)
plt.xlabel('Length of Stay (Days)')
plt.ylabel('Frequency')
plt.hist(length_of_stay, bins=30, color='blue', edgecolor='black')

# Create a histogram to visualize the distribution after taking IQR
plt.subplot(3, 1, 3)
plt.hist(filtered_data, bins=30, color='green', edgecolor='black', density=True)
plt.xlabel('Length of Stay (Days)')
plt.ylabel('Probability Density')
plt.title('Distribution after Taking Interquartile Range')

plt.tight_layout()  # Adjust layout to prevent overlapping

# Save the plot to a file in a different directory
# Get the absolute path of the current script's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Navigate to the parent directory and then enter Python_Generated_Images
output_directory = os.path.join(current_directory, '..', 'Python_Generated_Images')
output_filename = 'Simulated_Inpatient_Hospital_LOS_Gamma_distribution.png'
output_path = os.path.join(output_directory, output_filename)
plt.savefig(output_path)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate a dataset of LSAT scores
num_samples = 1000
lsat_scores = np.random.normal(loc=153, scale=10, size=num_samples)  # mean=152, std=10

# Adjust percentiles to match desired LSAT scores
percentiles = [10, 25, 40, 60, 70, 77, 97]
desired_scores = [140, 146, 150, 155, 158, 160, 170]

twenty_fifth_percentile = np.quantile(lsat_scores, 0.25)
seventy_fifth_percentile = np.quantile(lsat_scores, 0.75)

#Ignore the code below here:
try:
  print("The value that splits 25% of the data is " + str(twenty_fifth_percentile) + "\n")
except NameError:
  print("You haven't defined twenty_fifth_percentile.")

try:
  print("The value that splits 75% of the data is " + str(seventy_fifth_percentile) + "\n")
except NameError:
  print("You haven't defined seventy_fifth_percentile.")


# Define quartiles, deciles, and tenth here:
quartiles = np.quantile(lsat_scores, [0.25, 0.5, 0.75])
deciles = np.quantile(lsat_scores, list(np.arange(.1,1,.1)))

#Ignore the code below here:
try:
  print("The quartiles are " + str(quartiles) + "\n")
except NameError:
  print("You haven't defined quartiles.\n")
try:
  print("The deciles are " + str(deciles) + "\n")
except NameError:
  print("You haven't defined deciles.\n")
  
# np.searchsorted finds the index where this LSAT score should be inserted into the sorted array.
# np.insert serts the desired LSAT score at the calculated index, effectively adjusting the distribution to match the specified percentiles.
# This isn't needed
#for percentile, desired_score in zip(percentiles, desired_scores):
#    lsat_scores = np.insert(lsat_scores, np.searchsorted(lsat_scores, np.percentile(lsat_scores, percentile)), desired_score)

# Ensure all scores are within the range [120, 180]
lsat_scores = np.clip(lsat_scores, 120, 180)

# Calculate mean and dictionary of desired percentiles with corresponding LSAT scores
mean_score = np.mean(lsat_scores)
# dictionary comprehension to generate dictionary
percentiles_dict = {percentile: np.percentile(lsat_scores, percentile) for percentile in percentiles}

# Set the size and DPI of the figure
fig = plt.figure(figsize=(16, 10), dpi=100)  # Adjust the size and DPI as needed

# Create histogram
plt.hist(lsat_scores, bins=30, color='blue', edgecolor='black')

# Add mean line
plt.axvline(mean_score, color='red', linestyle='solid', linewidth=2, label='Mean')

# Add desired percentile lines
for percentile, lsat_score in percentiles_dict.items():
    plt.axvline(lsat_score, color='green', linestyle='dotted', linewidth=3)
    plt.text(lsat_score, plt.ylim()[1]+6, f'{percentile}th', rotation=0, va='bottom', ha='center', color='black')
    plt.text(lsat_score, plt.ylim()[1]+4, f'{int(lsat_score)}', rotation=0, va='top', ha='center', color='black')

# Set labels and title
plt.xlabel('LSAT Score')
plt.ylabel('Frequency')
plt.title('Distribution of LSAT Scores with Percentiles',  y=1.10) # y moves the title up to create more space

# Add legend
plt.legend()

# Save plot as image
plt.savefig('LSAT_score_percentiles.png', bbox_inches='tight')

# Show the plot
plt.show()

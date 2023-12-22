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

for percentile, desired_score in zip(percentiles, desired_scores):
    lsat_scores = np.insert(lsat_scores, np.searchsorted(lsat_scores, np.percentile(lsat_scores, percentile)), desired_score)

# Ensure all scores are within the range [120, 180]
lsat_scores = np.clip(lsat_scores, 120, 180)

# Calculate mean and updated deciles
mean_score = np.mean(lsat_scores)
deciles_dict = {percentile: np.percentile(lsat_scores, percentile) for percentile in percentiles}

# Set the size and DPI of the figure
fig = plt.figure(figsize=(16, 10), dpi=100)  # Adjust the size and DPI as needed

# Create histogram
plt.hist(lsat_scores, bins=30, color='blue', edgecolor='black')

# Add mean line
plt.axvline(mean_score, color='red', linestyle='solid', linewidth=2, label='Mean')

# Add updated decile lines
for percentile, decile in deciles_dict.items():
    plt.axvline(decile, color='green', linestyle='dotted', linewidth=3)
    plt.text(decile, plt.ylim()[1]+6, f'{percentile}th', rotation=0, va='bottom', ha='center', color='black')
    plt.text(decile, plt.ylim()[1]+4, f'{int(decile)}', rotation=0, va='top', ha='center', color='black')

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

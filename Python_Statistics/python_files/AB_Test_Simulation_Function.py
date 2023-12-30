import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# No random seed is set, so you should get a different result each time.
# We are simulating the proporation of rests that have a significant result given the parameters
def AB_Test_Simulator(significance_threshold = 0.05, sample_size = 1000, lift = .2, baseline_rate = .5):
    target_rate = (1 + lift) * baseline_rate
    results = [] # empty list
    # start the loop
    for i in range(100):
      # simulate data:
      sample_control = np.random.choice(['yes', 'no'],  size=int(sample_size/2), p=[baseline_rate, 1-baseline_rate])
      sample_name = np.random.choice(['yes', 'no'], size=int(sample_size/2), p=[target_rate, 1-target_rate])
      group = ['control']*int(sample_size/2) + ['name']*int(sample_size/2)
      outcome = list(sample_control) + list(sample_name)
      sim_data = {"Email": group, "Opened": outcome}
      sim_data = pd.DataFrame(sim_data)

      # run the test
      ab_contingency = pd.crosstab(np.array(sim_data.Email), np.array(sim_data.Opened))
      chi2, pval, dof, expected = chi2_contingency(ab_contingency)
      result = ('significant' if pval < significance_threshold else 'not significant')

      # append the result to our results list:
      results.append(result)
    return results

# calculate proportion of significant results:

results = AB_Test_Simulator(.05, 200, .3, .5)
print("Proportion of significant results:")
results =  np.array(results)
print(np.sum(results == 'significant')/100)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data=pd.read_excel('Controller Software.xlsx')
data.info()

data.sort_values(by='Day', inplace=True)


data['Cumulative_Defects'] = data['Defect'].cumsum()


t = data['Day'].values
cumulative_defects = data['Cumulative_Defects'].values

# Define the Goel-Okumoto model function based on the formula
def goel_okumoto(t, a, b):
    """
    Calculate the predicted cumulative defects at time t using the Goel-Okumoto model.

    Parameters:
    - t: Execution time or number of tests.
    - a: Expected number of total defects in the code.
    - b: Rate at which the failure rate decreases (Roundness factor).

    Returns:
    - Predicted cumulative defects at time t.
    """
    return a * (1 - np.exp(-b * t))

# Use curve fitting to estimate the parameters 'a' and 'b' for Goel-Okumoto
params_goel_okumoto, covariance_goel_okumoto = curve_fit(goel_okumoto, t, cumulative_defects, maxfev=10000)

# Extract the estimated parameters 'a' and 'b'
a_goel_okumoto, b_goel_okumoto = params_goel_okumoto

# Calculate the predicted cumulative defects using Goel-Okumoto
predicted_cumulative_defects_goel_okumoto = goel_okumoto(t, a_goel_okumoto, b_goel_okumoto)

# Calculate evaluation metrics for Goel-Okumoto
mae_goel_okumoto = mean_absolute_error(cumulative_defects, predicted_cumulative_defects_goel_okumoto)
rmse_goel_okumoto = np.sqrt(mean_squared_error(cumulative_defects, predicted_cumulative_defects_goel_okumoto))
r_squared_goel_okumoto = r2_score(cumulative_defects, predicted_cumulative_defects_goel_okumoto)


# Calculate the maximum MAE and RMSE values for Goel-Okumoto predictions
max_mae_goel_okumoto = np.max(np.abs(cumulative_defects - predicted_cumulative_defects_goel_okumoto))
max_rmse_goel_okumoto = np.max(np.sqrt((cumulative_defects - predicted_cumulative_defects_goel_okumoto) ** 2))

print("Maximum MAE (Goel-Okumoto):", max_mae_goel_okumoto)
print("Maximum RMSE (Goel-Okumoto):", max_rmse_goel_okumoto)


# Print evaluation metrics for Goel-Okumoto
print("Goel-Okumoto Model Evaluation:")
print("MAE:", mae_goel_okumoto)
print("RMSE:", rmse_goel_okumoto)
print("R-squared:", r_squared_goel_okumoto)



# Plot the observed data and model predictions for Goel-Okumoto
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t, cumulative_defects, label="Observed Data")
plt.plot(t, predicted_cumulative_defects_goel_okumoto, label="Goel-Okumoto Predictions")
plt.xlabel("Time (Days)")
plt.ylabel("Cumulative Defects")
plt.title("Goel-Okumoto Model")
plt.legend()

plt.tight_layout()
plt.show()
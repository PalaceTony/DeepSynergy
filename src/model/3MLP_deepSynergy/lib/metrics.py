import numpy as np
import scipy


# Function to calculate confidence interval
def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    n = len(data)
    stderr = np.std(data, ddof=1) / np.sqrt(n)
    margin = stderr * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return (mean, mean - margin, mean + margin)


# Provided MSE and RMSE values from logs
mse_values = np.array([217.8438, 333.6062, 284.0845, 224.2007, 213.4161])
rmse_values = np.array([14.7595, 18.2649, 16.8548, 14.9733, 14.6088])

# Calculate mean and confidence intervals
mean_mse, lower_mse, upper_mse = confidence_interval(mse_values)
mean_rmse, lower_rmse, upper_rmse = confidence_interval(rmse_values)

# Calculate standard deviation
std_mse = np.std(mse_values, ddof=1)
std_rmse = np.std(rmse_values, ddof=1)

# Report results similar to the figure
print("MSE: {:.2f}".format(mean_mse))
print("MSE 95% Confidence Interval: [{:.2f}, {:.2f}]".format(lower_mse, upper_mse))
print("RMSE: {:.2f} ± {:.2f}".format(mean_rmse, std_rmse))
print("RMSE 95% Confidence Interval: [{:.2f}, {:.2f}]".format(lower_rmse, upper_rmse))
print("RMSE Range: [{:.2f}, {:.2f}]".format(min(rmse_values), max(rmse_values)))

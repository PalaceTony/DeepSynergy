# Log data as a multiline string
log_data = """
2024-04-02 17:11:53,541 - 3MLP - Test MSE: 253.4249
2024-04-02 17:11:53,542 - 3MLP - Test RMSE: 15.9193
2024-04-02 17:11:53,542 - 3MLP - Test MAE: 10.6243
2024-04-02 17:11:53,542 - 3MLP - Test Pearson: 0.6512
2024-04-02 17:11:53,542 - 3MLP - Test Pearson p-value: 0.0000
2024-04-02 17:11:53,542 - 3MLP - Test Loss: 240.4802
2024-04-02 17:11:53,542 - 3MLP - Best validation loss: 259.2736

2024-04-02 17:23:53,210 - 3MLP - Test MSE: 279.1779
2024-04-02 17:23:53,210 - 3MLP - Test RMSE: 16.7086
2024-04-02 17:23:53,210 - 3MLP - Test MAE: 10.9892
2024-04-02 17:23:53,210 - 3MLP - Test Pearson: 0.6450
2024-04-02 17:23:53,210 - 3MLP - Test Pearson p-value: 0.0000
2024-04-02 17:23:53,210 - 3MLP - Test Loss: 303.0912
2024-04-02 17:23:53,210 - 3MLP - Best validation loss: 295.9915

2024-04-02 17:42:08,819 - 3MLP - Test MSE: 305.3448
2024-04-02 17:42:08,819 - 3MLP - Test RMSE: 17.4741
2024-04-02 17:42:08,819 - 3MLP - Test MAE: 10.8148
2024-04-02 17:42:08,819 - 3MLP - Test Pearson: 0.6712
2024-04-02 17:42:08,819 - 3MLP - Test Pearson p-value: 0.0000
2024-04-02 17:42:08,819 - 3MLP - Test Loss: 305.4253
2024-04-02 17:42:08,820 - 3MLP - Best validation loss: 380.9003

2024-04-02 17:57:29,389 - 3MLP - Test MSE: 392.6517
2024-04-02 17:57:29,390 - 3MLP - Test RMSE: 19.8154
2024-04-02 17:57:29,390 - 3MLP - Test MAE: 12.2798
2024-04-02 17:57:29,390 - 3MLP - Test Pearson: 0.6164
2024-04-02 17:57:29,390 - 3MLP - Test Pearson p-value: 0.0000
2024-04-02 17:57:29,390 - 3MLP - Test Loss: 372.2286
2024-04-02 17:57:29,390 - 3MLP - Best validation loss: 244.1967

2024-04-02 18:01:37,711 - 3MLP - Test MSE: 342.8251
2024-04-02 18:01:37,711 - 3MLP - Test RMSE: 18.5155
2024-04-02 18:01:37,711 - 3MLP - Test MAE: 12.1729
2024-04-02 18:01:37,711 - 3MLP - Test Pearson: 0.5933
2024-04-02 18:01:37,711 - 3MLP - Test Pearson p-value: 0.0000
2024-04-02 18:01:37,711 - 3MLP - Test Loss: 361.2523
2024-04-02 18:01:37,711 - 3MLP - Best validation loss: 301.6772
"""

# Split the data into lines
lines = log_data.strip().split("\n")

# Initialize dictionaries to hold sum of metrics and their counts
metrics_sums = {
    "MSE": 0,
    "RMSE": 0,
    "MAE": 0,
    "Pearson": 0,
    "Pearson p-value": 0,
    "Test Loss": 0,
    "Best validation loss": 0,
}
counts = {key: 0 for key in metrics_sums.keys()}

# Extract metrics from each line, sum them, and count the occurrences
for line in lines:
    parts = line.split(" - ")
    metric_info = parts[-1].split(": ")
    if len(metric_info) == 2:  # Ensures the line is in the correct format
        metric_name = metric_info[0].replace("Test ", "")
        metric_value = float(metric_info[1])
        if metric_name in metrics_sums:
            metrics_sums[metric_name] += metric_value
            counts[metric_name] += 1

# Calculate averages for each metric
averages = {}
for metric, total in metrics_sums.items():
    if counts[metric] > 0:  # Ensures division by zero does not occur
        averages[metric] = total / counts[metric]

# Display the average values
for metric, average in averages.items():
    print(f"{metric}: {average:.4f}")

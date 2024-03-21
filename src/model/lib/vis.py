import matplotlib.pyplot as plt


def plot_performance(val_losses, smooth_val_loss, average_test_loss):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(val_losses, label="Validation Loss during Hyperparameter Selection")
    ax.plot(smooth_val_loss, label="Smoothed Validation Loss")
    ax.axhline(y=average_test_loss, color="r", linestyle="-", label="Average Test Loss")
    ax.legend()
    plt.savefig("performance.png")

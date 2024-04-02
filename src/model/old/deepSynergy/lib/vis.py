import matplotlib.pyplot as plt


def plot_performance(val_losses, smooth_val_loss, test_loss):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(val_losses, label="Validation Loss during Hyperparameter Selection")
    ax.plot(smooth_val_loss, label="Smoothed Validation Loss")
    # Assuming test_loss values should be plotted against their index
    test_loss_indices = range(len(test_loss))
    ax.plot(test_loss_indices, test_loss, "ro-", label="Test Loss")
    ax.legend()
    plt.savefig("performance.png")

import torch

def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits data into train, validation, and test sets.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

    num_samples = data.size(0)
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    test_size = num_samples - train_size - val_size

    train_data, val_data, test_data = torch.utils.data.random_split(
        data, [train_size, val_size, test_size]
    )

    return train_data, val_data, test_data

def normalize_station_data(station_data):
    """
    Normalize station data using Min-Max Normalization.

    Args:
        station_data: Tensor of shape (num_samples, 2), where each row is a 2D station coordinate.

    Returns:
        Normalized station data: Tensor of the same shape as input, with values in range [0, 1].
    """
    min_vals = station_data.min(dim=0, keepdim=True).values  # Min value for each column (x, y)
    max_vals = station_data.max(dim=0, keepdim=True).values  # Max value for each column (x, y)
    normalized_data = (station_data - min_vals) / (max_vals - min_vals)
    return normalized_data


import os
import matplotlib.pyplot as plt

def plot_patches(stage, 
                 original=None, 
                 masked=None, 
                 reconstructed=None, 
                 epoch=None, 
                 input_data=None, 
                 actual_data=None, 
                 predicted_data=None, 
                 sample_idx=0, 
                 num_patches=5, 
                 note=None, 
                 title="Reconstructed Patches", 
                 save_dir=None, 
                 forecast_dir=None, 
                 patches_dir=None,
                 SAMPLE_LENGTH=None):
    """
    Unified plotting function for different stages of the model.
    
    Args:
        stage: The stage of the model (1, 2, or 3).
        original, masked, reconstructed: Used for stage 1 (patch reconstruction).
        epoch: Current epoch number (used for stage 2 and 3).
        input_data, actual_data, predicted_data: Used for stage 2 and 3 (forecasting).
        sample_idx: Index of the sample to plot (default: 0).
        num_patches: Number of patches to plot for stage 1 (default: 5).
        note: Additional note for saving the plot.
        title: Title of the plot (default: "Reconstructed Patches").
    """
    if stage == 1:
        # Plot reconstructed patches (Stage 1)
        fig, axes = plt.subplots(num_patches, 3, figsize=(12, num_patches * 3))
        fig.suptitle(title, fontsize=16)

        for i in range(num_patches):
            axes[i, 0].plot(original[i].cpu().numpy())
            axes[i, 0].set_title("Original")
            axes[i, 1].plot(masked[i].cpu().numpy())
            axes[i, 1].set_title("Masked")
            axes[i, 2].plot(reconstructed[i].cpu().numpy())
            axes[i, 2].set_title("Reconstructed")

        plt.tight_layout()
        save_path = os.path.join(save_dir, patches_dir, title + ".png")
        print("saved the plot to:", save_path)
    
    elif stage in [2, 3]:
        # Plot forecasted patches (Stage 2 and 3)
        num_channels = input_data.shape[1]
        sample_input = input_data[sample_idx].cpu().numpy()
        sample_actual = actual_data[sample_idx].cpu().numpy()
        sample_predicted = predicted_data[sample_idx].cpu().numpy()
        predicted_continuous = sample_predicted.reshape(-1)[:SAMPLE_LENGTH]

        plt.figure(figsize=(15, 5 * num_channels))

        for channel in range(num_channels):
            plt.subplot(num_channels, 1, channel + 1)
            plt.plot(range(SAMPLE_LENGTH), sample_input[channel], label="Input", color='blue')
            plt.plot(range(SAMPLE_LENGTH, SAMPLE_LENGTH * 2), sample_actual[channel], label="Actual", color='green')
            plt.plot(range(SAMPLE_LENGTH, SAMPLE_LENGTH * 2), predicted_continuous, label="Forecasted", color='red')
            plt.title(f"Channel {channel + 1} - Epoch {epoch}")
            plt.legend()

        plt.tight_layout()
        filename = f"forecast_epoch_{epoch}.png"
        if note:
            filename = f"forecast_epoch_{epoch}_{note}.png"
        save_path = os.path.join(save_dir, forecast_dir, filename)
    
    # Save and close the plot
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()

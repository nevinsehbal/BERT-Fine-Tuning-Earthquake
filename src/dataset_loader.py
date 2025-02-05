import pandas as pd
import os
import numpy as np
import torch
import json


def load_sinusoidal_samples_from_csv(sample_length):
    """
    Load sinusoidal samples and station information from a CSV and return as a PyTorch tensor.
    Args:
        sample_length (int): Length of each time-series sample.
    Returns:
        tuple: (data_tensor, station_tensor)
            - data_tensor: PyTorch tensor of shape (num_samples, num_channels, sample_length).
            - station_tensor: PyTorch tensor of station IDs of shape (num_samples,).
    """
    dataset_dir = "sinusoidal_dataset"
    csv_path = os.path.join(dataset_dir, "sinusoidal_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist. Set READ_FROM_CSV = False to generate the dataset.")
    
    # Read the dataset
    data_df = pd.read_csv(csv_path)
    data_df['Station'] = data_df['Station'].apply(lambda x: eval(x))
    
    # Extract the channel data and convert to lists
    data = []
    stations = []
    for _, row in data_df.iterrows():
        channels = [
            json.loads(row['Channel_1'])[:sample_length],
            json.loads(row['Channel_2'])[:sample_length],
            json.loads(row['Channel_3'])[:sample_length]
        ]
        data.append(channels)
        stations.append(row['Station'])  # Collect station info
    
    # Convert to numpy arrays
    data_array = np.array(data)  # Shape: (num_samples, num_channels, sample_length)
    station_array = np.array(stations)  # Shape: (num_samples,)

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data_array, dtype=torch.float32)
    station_tensor = torch.tensor(station_array, dtype=torch.long)  # Assuming station IDs are integers

    return data_tensor, station_tensor


def load_stage2_data_from_csv(sample_length, n_future_patches, patch_size):
    """
    Load data for stage 2 training from a CSV file, and generate future patches by splitting samples.

    Args:
        sample_length (int): Length of each time-series sample (past context).
        n_future_patches (int): Number of future patches to predict.
        patch_size (int): Size of each patch.

    Returns:
        tuple: (data_tensor, station_tensor, future_patches_tensor)
            - data_tensor: PyTorch tensor of shape (num_samples, num_channels, sample_length).
            - station_tensor: PyTorch tensor of station IDs of shape (num_samples,).
            - future_patches_tensor: PyTorch tensor of shape (num_samples, num_channels * n_future_patches, patch_size).
    """
    dataset_dir = "sinusoidal_dataset"
    csv_path = os.path.join(dataset_dir, "sinusoidal_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist. Ensure the dataset is prepared for stage 2.")

    # Read the dataset
    data_df = pd.read_csv(csv_path)
    data_df['Station'] = data_df['Station'].apply(lambda x: eval(x))  # Parse station data if stored as a list

    # Define arrays to store the data
    data = []
    stations = []
    future_patches = []

    # Split each row into past (context) and future (predicted patches)
    for _, row in data_df.iterrows():
        # Extract the long sequence for each channel
        long_sequence = [
            json.loads(row['Channel_1']),
            json.loads(row['Channel_2']),
            json.loads(row['Channel_3'])
        ]
        station = row['Station']  # Extract station information

        # Ensure the sequence is long enough
        total_length_required = sample_length + (n_future_patches * patch_size)
        if len(long_sequence[0]) < total_length_required:
            raise ValueError(f"Sample length is too short for splitting: {len(long_sequence[0])} < {total_length_required}")

        # Split into past (context) and future
        past_context = [seq[:sample_length] for seq in long_sequence]  # Shape: (num_channels, sample_length)
        future_context = [
            np.array(seq[sample_length:sample_length + (n_future_patches * patch_size)])
            for seq in long_sequence
        ]  # Shape: (num_channels, n_future_patches * patch_size)

        # Reshape future patches into individual patches
        future_patches_flattened = [
            seq.reshape(n_future_patches, patch_size).flatten() for seq in future_context
        ]  # Shape: (num_channels * n_future_patches, patch_size)

        # Append to the respective lists
        data.append(past_context)
        stations.append(station)
        future_patches.append(np.concatenate(future_patches_flattened, axis=0))  # Flattened future patches

    # Convert to numpy arrays
    data_array = np.array(data)  # Shape: (num_samples, num_channels, sample_length)
    station_array = np.array(stations)  # Shape: (num_samples,)
    future_patches_array = np.array(future_patches)  # Shape: (num_samples, num_channels * n_future_patches * patch_size)

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data_array, dtype=torch.float32)
    station_tensor = torch.tensor(station_array, dtype=torch.long)  # Assuming station IDs are integers
    future_patches_tensor = torch.tensor(future_patches_array, dtype=torch.float32)

    return data_tensor, station_tensor, future_patches_tensor


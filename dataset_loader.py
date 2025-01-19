import pandas as pd
import os
import numpy as np

def get_dataframe():
    dataset_dir = "sinusoidal_dataset"
    # Read the dataset
    csv_path = os.path.join(dataset_dir, "sinusoidal_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist. Set READ_FROM_CSV = False to generate the dataset.")
    # Read the CSV file
    data_df = pd.read_csv(csv_path)
    print("data head: ", data_df.head())
    data_df['Station'] = data_df['Station'].apply(lambda x: eval(x))
    return data_df

import torch
import pandas as pd
import os
import json

def load_sinusoidal_samples_from_csv(sample_length):
    """
    Load sinusoidal samples from a CSV and return as a PyTorch tensor.

    Args:
        sample_length (int): Length of each time-series sample.

    Returns:
        torch.Tensor: Loaded data of shape (num_samples, num_channels, sample_length).
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
    for _, row in data_df.iterrows():
        channels = [
            json.loads(row['Channel_1'])[:sample_length],
            json.loads(row['Channel_2'])[:sample_length],
            json.loads(row['Channel_3'])[:sample_length]
        ]
        data.append(channels)
    
    # Convert to numpy array with shape (num_samples, num_channels, sample_length)
    data_array = np.array(data)
    
    # Convert to PyTorch tensor
    return torch.tensor(data_array, dtype=torch.float32)

# # Example usage
# data_tensor = load_sinusoidal_samples_from_csv(sample_length=60000)
# print(data_tensor.shape)  # Expecting shape: (num_samples, 3, sample_length)
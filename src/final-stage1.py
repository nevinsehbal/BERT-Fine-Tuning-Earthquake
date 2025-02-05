import torch
import dataset_loader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import time
import random
import numpy as np
from models import TimeSeriesBERT
from utils import split_dataset, normalize_station_data, plot_patches

save_dir = "logs_first_stage"
patches_dir = "patches"+str(time.time())
# Constants
NUM_SAMPLES = 1000
NUM_STATIONS = 6
NUM_CHANNELS = 3
SAMPLE_LENGTH = 600
PATCH_SIZE = 50
NUM_PATCHES = SAMPLE_LENGTH // PATCH_SIZE
EMBED_DIM = 768  # Embedding dimension for BERT
PATCHES_PER_SAMPLE = NUM_PATCHES * NUM_CHANNELS 
BATCH_SIZE = 4
MASK_RATIO = 0.2  # 20% masking
LEARNING_RATE = 1e-4
EPOCHS = 30
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, patches_dir), exist_ok=True)

def train_model_with_plot(
    model, train_loader, val_loader, test_loader, epochs, learning_rate, mask_ratio, num_patches_to_plot=5
):
    """
    Train the model and plot reconstructed patches at each epoch.
    """
    optimizer = optim.AdamW([
        {'params': model.bert.parameters(), 'lr': learning_rate * 0.1},
        {'params': model.output_layer.parameters(), 'lr': learning_rate},
        {'params': model.embeddings.parameters(), 'lr': learning_rate},
    ])
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            data, station_ids = batch[0], batch[1]

            optimizer.zero_grad()

            # Mask patches
            batch_size, num_channels, sample_length = data.shape
            num_patches = sample_length // PATCH_SIZE
            mask = (torch.rand(batch_size, num_patches * num_channels) < mask_ratio).to(data.device)

            # Apply masking
            data_unfolded = data.unfold(-1, PATCH_SIZE, PATCH_SIZE).permute(0, 2, 1, 3)
            masked_data = data_unfolded.clone().reshape(batch_size, -1, PATCH_SIZE)
            masked_data[mask] = 0

            # Forward pass
            output = model(data, station_ids)
            target = data_unfolded.reshape(batch_size, -1, PATCH_SIZE)

            loss = criterion(output[mask], target[mask])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
        train_losses.append(epoch_loss / len(train_loader))

        # Evaluate on validation set
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                data, station_ids = batch[0], batch[1]

                # Mask patches
                batch_size, num_channels, sample_length = data.shape
                num_patches = sample_length // PATCH_SIZE
                mask = (torch.rand(batch_size, num_patches * num_channels) < mask_ratio).to(data.device)

                # Apply masking
                data_unfolded = data.unfold(-1, PATCH_SIZE, PATCH_SIZE).permute(0, 2, 1, 3)
                masked_data = data_unfolded.clone().reshape(batch_size, -1, PATCH_SIZE)
                masked_data[mask] = 0

                # Forward pass
                output = model(data, station_ids)
                target = data_unfolded.reshape(batch_size, -1, PATCH_SIZE)

                loss = criterion(output[mask], target[mask])
                epoch_val_loss += loss.item()

        val_losses.append(epoch_val_loss / len(val_loader))
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {epoch_val_loss / len(val_loader):.4f}")

        # Evaluation and plotting
        model.eval()
        with torch.no_grad():
            for loader, dataset_name in zip([train_loader, val_loader, test_loader], ["Train", "Val", "Test"]):
                station_wise_results = {}  # Dictionary to store station-specific results

                for batch in loader:
                    data, stations = batch[0], batch[1]  # Assuming batch[1] contains station data
                    data_unfolded = data.unfold(-1, PATCH_SIZE, PATCH_SIZE).permute(0, 2, 1, 3)
                    masked_data = data_unfolded.clone().reshape(data.size(0), -1, PATCH_SIZE)

                    # Apply masking
                    mask = (torch.rand(data.size(0), masked_data.size(1)) < mask_ratio).to(data.device)
                    masked_data[mask] = 0

                    # Model forward pass
                    output = model(data, stations)

                    # Group results by station
                    for i, station in enumerate(stations):
                        station_key = tuple(station.tolist())  # Convert station tensor to a hashable tuple
                        if station_key not in station_wise_results:
                            station_wise_results[station_key] = {"original": [], "reconstructed": [], "masked": []}

                        station_wise_results[station_key]["original"].append(
                            data_unfolded[i].reshape(-1, PATCH_SIZE)
                        )
                        station_wise_results[station_key]["reconstructed"].append(
                            output[i].reshape(-1, PATCH_SIZE)
                        )
                        station_wise_results[station_key]["masked"].append(
                            masked_data[i].reshape(-1, PATCH_SIZE)
                        )

                    # Plot a sample of reconstructions (first batch only)
                    plot_patches(
                        stage=1,
                        original=data_unfolded.reshape(-1, PATCH_SIZE)[:num_patches_to_plot],
                        masked=masked_data.reshape(-1, PATCH_SIZE)[:num_patches_to_plot],
                        reconstructed=output.reshape(-1, PATCH_SIZE)[:num_patches_to_plot],
                        save_dir=save_dir,
                        patches_dir=patches_dir,
                        title=f"{dataset_name} Set Reconstructed Patches (Epoch {epoch + 1})",
                    )
                    break  # Only plot for the first batch of each dataset

    return train_losses, val_losses

# Load both data and station data
data, station_data = dataset_loader.load_sinusoidal_samples_from_csv(sample_length=SAMPLE_LENGTH)
# Normalize station data
station_data = normalize_station_data(station_data)

# Split both data and station data into train, val, and test sets
train_data_subset, val_data_subset, test_data_subset = split_dataset(data)
train_station_subset, val_station_subset, test_station_subset = split_dataset(station_data)

# Extract tensors from Subset objects
train_data = torch.stack([data[i] for i in train_data_subset.indices])
val_data = torch.stack([data[i] for i in val_data_subset.indices])
test_data = torch.stack([data[i] for i in test_data_subset.indices])

train_station = torch.stack([station_data[i] for i in train_station_subset.indices])
val_station = torch.stack([station_data[i] for i in val_station_subset.indices])
test_station = torch.stack([station_data[i] for i in test_station_subset.indices])

# Create DataLoaders
train_loader = DataLoader(
    TensorDataset(train_data, train_station), 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    generator=torch.Generator().manual_seed(42)
)
val_loader = DataLoader(
    TensorDataset(val_data, val_station), 
    batch_size=BATCH_SIZE
)
test_loader = DataLoader(
    TensorDataset(test_data, test_station), 
    batch_size=BATCH_SIZE
)

# Initialize model
model = TimeSeriesBERT(EMBED_DIM, PATCH_SIZE, SAMPLE_LENGTH, NUM_CHANNELS, NUM_STATIONS, stage=1)

# Train the model with plotting
train_losses, val_losses = train_model_with_plot(
    model, train_loader, val_loader, test_loader, EPOCHS, LEARNING_RATE, MASK_RATIO, num_patches_to_plot=5
)

# Save the model
torch.save(model.state_dict(), os.path.join(save_dir, patches_dir, "final_model.pt"))

# save the losses
np.save(os.path.join(save_dir, patches_dir, "train_losses.npy"), train_losses)
np.save(os.path.join(save_dir, patches_dir, "val_losses.npy"), val_losses)

# Plot the losses and save
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_dir, patches_dir, "losses.png"))
plt.close()
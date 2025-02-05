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

save_dir = 'logs_second_stage'
forecast_dir = 'forecasts'+str(int(time.time()))
first_stage_model_path = '/home/hekimoglu/workspace/git/final/BERT-Fine-Tuning-Earthquake/logs/patches1737315105.74911/final_model.pt'
EMBED_DIM = 768  # Embedding dimension for BERT
PATCH_SIZE = 50
SAMPLE_LENGTH = 600
NUM_CHANNELS = 3
NUM_STATIONS = 6
EPOCHS = 50
BATCH_SIZE = 4
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, forecast_dir), exist_ok=True)

# Load the saved model state dictionary
model = TimeSeriesBERT(EMBED_DIM, PATCH_SIZE, SAMPLE_LENGTH, NUM_CHANNELS, NUM_STATIONS, stage=2)
model.load_state_dict(torch.load(first_stage_model_path))
# Modify the output layer (example: change output size to 64)
new_output_layer = nn.Linear(model.bert.config.hidden_size, PATCH_SIZE)
model.output_layer = new_output_layer
# Freeze the CustomEmbedding and Bert layers
for param in model.embeddings.parameters():
    param.requires_grad = False

for param in model.bert.parameters():
    param.requires_grad = False

# Reinitialize optimizer for unfrozen layers
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# Load both data and station data
data, station_data = dataset_loader.load_sinusoidal_samples_from_csv(sample_length=1200) # 600 for input and 600 for output
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

val_data_loader = DataLoader(
    TensorDataset(val_data, val_station), 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

test_data_loader = DataLoader(
    TensorDataset(test_data, test_station), 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

# Training loop
criterion = nn.MSELoss()
train_losses = []
val_losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch in train_loader:
        data, station_ids = batch[0], batch[1]

        # Split into input and output
        input_data = data[:, :, :SAMPLE_LENGTH]  # Input: First half
        target_data = data[:, :, SAMPLE_LENGTH:]  # Target: Second half

        # Prepare input for the model
        unfolded_input = input_data.unfold(-1, PATCH_SIZE, PATCH_SIZE)
        unfolded_input = unfolded_input.permute(0, 2, 1, 3).reshape(unfolded_input.shape[0], -1, PATCH_SIZE)

        # Model forward pass
        output = model(unfolded_input, station_ids)

        # Adjust output shape if needed
        unfolded_target = target_data.unfold(-1, PATCH_SIZE, PATCH_SIZE)
        unfolded_target = unfolded_target.permute(0, 2, 1, 3).reshape(unfolded_target.shape[0], -1, PATCH_SIZE)

        loss = criterion(output, unfolded_target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(train_loader):.4f}")
    train_losses.append(epoch_loss / len(train_loader))
    
    # validate model
    validate_epoch_loss = 0.0
    val_predictions = []
    for val_batch in val_data_loader:
        val_data, val_station_ids = val_batch[0], val_batch[1]
        val_input_data = val_data[:, :, :SAMPLE_LENGTH]  # Input: First half
        val_target_data = val_data[:, :, SAMPLE_LENGTH:]  # Target: Second half

        # Prepare input for the model
        unfolded_val_input = val_input_data.unfold(-1, PATCH_SIZE, PATCH_SIZE)
        unfolded_val_input = unfolded_val_input.permute(0, 2, 1, 3).reshape(unfolded_val_input.shape[0], -1, PATCH_SIZE)

        # Model forward pass
        val_output = model(unfolded_val_input, val_station_ids)

        # Adjust output shape if needed
        unfolded_val_target = val_target_data.unfold(-1, PATCH_SIZE, PATCH_SIZE)
        unfolded_val_target = unfolded_val_target.permute(0, 2, 1, 3).reshape(unfolded_val_target.shape[0], -1, PATCH_SIZE)

        val_loss = criterion(val_output, unfolded_val_target)
        validate_epoch_loss += val_loss.item()
        # Store for visualization (only first batch)
        if len(val_predictions) == 0:
            val_predictions = val_output.detach()
            val_inputs = val_input_data.detach()
            val_actuals = val_target_data.detach()

    print(f"Validation Loss: {validate_epoch_loss / len(val_data_loader):.4f}")
    val_losses.append(validate_epoch_loss / len(val_data_loader))

    # # Plot predictions for the first batch of validation data
    # plot_patches(epoch + 1, val_inputs, val_actuals, val_predictions, sample_idx=0)
    plot_patches(
    stage=2,
    epoch=epoch + 1,
    input_data=val_inputs,
    actual_data=val_actuals,
    predicted_data=val_predictions,
    sample_idx=0,  # Optional: Specify which sample to plot
    save_dir=save_dir,
    forecast_dir=forecast_dir,
    SAMPLE_LENGTH=SAMPLE_LENGTH
    )


# Save the model
torch.save(model.state_dict(), os.path.join(save_dir, forecast_dir, 'final_model.pt'))

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(save_dir, forecast_dir, 'losses.png'))
plt.close()

# Test the model

test_losses = []
test_predictions = []

for test_batch in test_data_loader:
    test_data, test_station_ids = test_batch[0], test_batch[1]
    test_input_data = test_data[:, :, :SAMPLE_LENGTH]  # Input: First half
    test_target_data = test_data[:, :, SAMPLE_LENGTH:]  # Target: Second half

    # Prepare input for the model
    unfolded_test_input = test_input_data.unfold(-1, PATCH_SIZE, PATCH_SIZE)
    unfolded_test_input = unfolded_test_input.permute(0, 2, 1, 3).reshape(unfolded_test_input.shape[0], -1, PATCH_SIZE)

    # Model forward pass
    test_output = model(unfolded_test_input, test_station_ids)

    # Adjust output shape if needed
    unfolded_test_target = test_target_data.unfold(-1, PATCH_SIZE, PATCH_SIZE)
    unfolded_test_target = unfolded_test_target.permute(0, 2, 1, 3).reshape(unfolded_test_target.shape[0], -1, PATCH_SIZE)

    test_loss = criterion(test_output, unfolded_test_target)
    test_losses.append(test_loss.item())

    # Store for visualization (only first batch)
    if len(test_predictions) == 0:
        test_predictions = test_output.detach()
        test_inputs = test_input_data.detach()
        test_actuals = test_target_data.detach()

print(f"Test Loss: {sum(test_losses) / len(test_data_loader):.4f}")

# Plot predictions for the first batch of test data

plot_patches(0, test_inputs, test_actuals, test_predictions, sample_idx=0, note='test')
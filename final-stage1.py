import dataset_loader
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertConfig
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import time

# Constants
NUM_SAMPLES = 1000
NUM_STATIONS = 5
NUM_CHANNELS = 3
SAMPLE_LENGTH = 600
PATCH_SIZE = 50
NUM_PATCHES = SAMPLE_LENGTH // PATCH_SIZE
EMBED_DIM = 768  # Embedding dimension for BERT
PATCHES_PER_SAMPLE = NUM_PATCHES * NUM_CHANNELS 
BATCH_SIZE = 4
MASK_RATIO = 0.2  # 20% masking
LEARNING_RATE = 1e-4
EPOCHS = 5

save_dir = "logs"
patches_dir = "patches"+str(time.time())
os.makedirs(save_dir, exist_ok=True)

class CustomEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, sample_length):
        super(CustomEmbedding, self).__init__()
        self.patch_embedding = nn.Linear(patch_size, embed_dim)
        self.time_weight = nn.Parameter(torch.randn(2, embed_dim))  # Trainable W_t matrix
        self.sample_length = sample_length
        self.patch_size = patch_size
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_channels, sample_length).
        Returns:
            Tensor of shape (batch_size, num_patches, embed_dim).
        """
        batch_size, num_channels, sample_length = x.shape

        # Split into patches: shape becomes (batch_size, num_channels * num_patches, patch_size)
        num_patches = sample_length // self.patch_size
        patches = x.unfold(-1, self.patch_size, self.patch_size)  # (batch_size, num_channels, num_patches, patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(batch_size, num_patches * num_channels, self.patch_size)

        # Patch embedding
        patch_embeds = self.patch_embedding(patches)  # (batch_size, num_patches * num_channels, embed_dim)

        # Time embedding: Calculate normalized sine and cosine for each patch index
        patch_indices = torch.arange(0, num_patches * num_channels).to(x.device)  # (num_patches * num_channels)
        time_encodings = torch.cat([
            torch.sin(2 * torch.pi * patch_indices / self.sample_length).unsqueeze(-1),
            torch.cos(2 * torch.pi * patch_indices / self.sample_length).unsqueeze(-1)
        ], dim=-1)  # Shape: (num_patches * num_channels, 2)
        time_embeds = time_encodings @ self.time_weight  # (num_patches * num_channels, embed_dim)
        time_embeds = time_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # Match batch size

        # Combine embeddings
        return patch_embeds + time_embeds

class TimeSeriesBERT(nn.Module):
    def __init__(self, embed_dim, patch_size, sample_length, num_channels):
        super(TimeSeriesBERT, self).__init__()
        self.embeddings = CustomEmbedding(embed_dim, patch_size, sample_length)
        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Adjust the output layer to match time-series patch size
        self.output_layer = nn.Linear(self.bert.config.hidden_size, patch_size)
        self.num_channels = num_channels

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_channels, sample_length).
        Returns:
            Tensor of shape (batch_size, num_patches, patch_size).
        """
        embeds = self.embeddings(x)  # (batch_size, num_patches * num_channels, embed_dim)
        batch_size, seq_len, embed_dim = embeds.shape

        bert_output = self.bert(inputs_embeds=embeds).last_hidden_state  # (batch_size, seq_len, embed_dim)
        output = self.output_layer(bert_output)  # (batch_size, seq_len, patch_size)

        # Reshape to separate patches: (batch_size, num_patches, patch_size)
        num_patches = seq_len // x.size(1)  # Divide by num_channels
        return output.view(batch_size, num_patches*self.num_channels, self.output_layer.out_features)

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

def plot_reconstructed_patches(original, reconstructed, masked, num_patches=5, title="Reconstructed Patches"):
    """
    Plot original, masked, and reconstructed patches.
    """
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
    # plt.show()
    # do not show, instead save to a directory named bert4st_runs
    # create save_dir/patches_dir if not exists
    if not os.path.exists(save_dir + '/' + patches_dir):
        os.makedirs(save_dir + '/' + patches_dir)
    plt.savefig(os.path.join(save_dir, patches_dir, title + ".png"))
    plt.close()

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

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            data = batch[0]

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
            output = model(data)
            target = data_unfolded.reshape(batch_size, -1, PATCH_SIZE)

            loss = criterion(output[mask], target[mask])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        # Evaluation and plotting
        model.eval()
        with torch.no_grad():
            for loader, dataset_name in zip([train_loader, val_loader, test_loader], ["Train", "Val", "Test"]):
                for batch in loader:
                    data = batch[0]
                    data_unfolded = data.unfold(-1, PATCH_SIZE, PATCH_SIZE).permute(0, 2, 1, 3)
                    masked_data = data_unfolded.clone().reshape(data.size(0), -1, PATCH_SIZE)
                    mask = (torch.rand(data.size(0), masked_data.size(1)) < mask_ratio).to(data.device)
                    masked_data[mask] = 0

                    output = model(data)
                    plot_reconstructed_patches(
                        data_unfolded.reshape(-1, PATCH_SIZE)[:num_patches_to_plot],
                        output.reshape(-1, PATCH_SIZE)[:num_patches_to_plot],
                        masked_data.reshape(-1, PATCH_SIZE)[:num_patches_to_plot],
                        # data_unfolded.view(-1, PATCH_SIZE)[:num_patches_to_plot],
                        # output.view(-1, PATCH_SIZE)[:num_patches_to_plot],
                        # masked_data.view(-1, PATCH_SIZE)[:num_patches_to_plot],
                        title=f"{dataset_name} Set Reconstructed Patches (Epoch {epoch + 1})"
                    )
                    break  # Only plot for the first batch of each dataset

data = dataset_loader.load_sinusoidal_samples_from_csv(sample_length=SAMPLE_LENGTH)

train_data, val_data, test_data = split_dataset(data)

# Extract tensors from Subset objects
train_data = torch.stack([data[i] for i in train_data.indices])
val_data = torch.stack([data[i] for i in val_data.indices])
test_data = torch.stack([data[i] for i in test_data.indices])

# Create DataLoaders
train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(test_data), batch_size=BATCH_SIZE)

# Initialize model
model = TimeSeriesBERT(EMBED_DIM, PATCH_SIZE, SAMPLE_LENGTH, NUM_CHANNELS)

# Train the model with plotting
train_model_with_plot(
    model, train_loader, val_loader, test_loader, EPOCHS, LEARNING_RATE, MASK_RATIO, num_patches_to_plot=5
)
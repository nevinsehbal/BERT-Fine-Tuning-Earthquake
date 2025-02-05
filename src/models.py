from transformers import BertModel, BertConfig
import torch.nn as nn
import torch

class CustomEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, sample_length, num_stations, station_embed_dim=64):
        super(CustomEmbedding, self).__init__()
        self.patch_embedding = nn.Linear(patch_size, embed_dim)
        self.station_embedding = nn.Linear(2, embed_dim)  # Project station IDs into the same dimension as embed_dim
        self.time_weight = nn.Parameter(torch.randn(2, embed_dim))  # Trainable W_t matrix
        self.sample_length = sample_length
        self.patch_size = patch_size
        self.embed_dim = embed_dim
    
    def forward(self, x, station_ids):
        """
        Args:
            x: Input tensor of shape (batch_size, num_channels, sample_length).
            station_ids: Station ID tensor of shape (batch_size, 2).
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

        # Time embedding
        patch_indices = torch.arange(0, num_patches * num_channels).to(x.device)
        time_encodings = torch.cat([
            torch.sin(2 * torch.pi * patch_indices / self.sample_length).unsqueeze(-1),
            torch.cos(2 * torch.pi * patch_indices / self.sample_length).unsqueeze(-1)
        ], dim=-1)  # Shape: (num_patches * num_channels, 2)
        time_embeds = time_encodings @ self.time_weight  # (num_patches * num_channels, embed_dim)
        time_embeds = time_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        # Station embedding
        station_embeds = self.station_embedding(station_ids)  # (batch_size, embed_dim)
        station_embeds = station_embeds.unsqueeze(1).expand(-1, num_patches * num_channels, -1)  # Match patch dims

        # Combine embeddings
        combined_embeds = patch_embeds + time_embeds + station_embeds  # Element-wise sum of all embeddings
        return combined_embeds
    
class TimeSeriesBERT(nn.Module):
    def __init__(self, embed_dim, patch_size, sample_length, num_channels, num_stations, stage=1):
        super(TimeSeriesBERT, self).__init__()
        self.embeddings = CustomEmbedding(embed_dim, patch_size, sample_length, num_stations)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.output_layer = nn.Linear(self.bert.config.hidden_size, patch_size)
        self.num_channels = num_channels
        self.sample_length = sample_length
        self.patch_size = patch_size
        self.stage = stage

    def forward(self, x, station_ids):
        """
        Args:
            x: Input tensor of shape (batch_size, num_channels, sample_length).
            station_ids: Tensor of station IDs (batch_size,).
        Returns:
            Tensor of shape (batch_size, num_patches, patch_size).
        """
        embeds = self.embeddings(x, station_ids)
        batch_size, seq_len, embed_dim = embeds.shape

        bert_output = self.bert(inputs_embeds=embeds).last_hidden_state  # (batch_size, seq_len, embed_dim)
        output = self.output_layer(bert_output)  # (batch_size, seq_len, patch_size)

        # Handle output differently depending on the stage
        if self.stage == 1:
            num_patches = seq_len // x.size(1)  # Stage 1: Divide by num_channels
        elif self.stage == 2:
            num_patches = self.sample_length // self.patch_size  # Stage 2: Based on sample_length and patch_size
        elif self.stage == 3:
            num_patches = self.sample_length // self.patch_size  # Stage 3: Same as Stage 2 but could have other custom logic

        return output.view(batch_size, num_patches * self.num_channels, self.output_layer.out_features)
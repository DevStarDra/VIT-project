
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets


class VisionTransformer(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, embedding_dim, num_layers, num_heads, mlp_ratio):
        super(VisionTransformer, self).__init__()
        self.patch_embed = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        input_size = 224  # Assumes input image size is 224 x 224
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=embedding_dim*mlp_ratio), num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # (batch_size, embedding_dim, num_patches, num_patches)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embedding_dim)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (batch_size, 1, embedding_dim)
        x = torch.cat([cls_token, x], dim=1)  # (batch_size, num_patches+1, embedding_dim)
        x += self.pos_embed[:, :x.size(1), :]  # (batch_size, num_patches+1, embedding_dim)
        x = self.transformer(x)  # (batch_size, num_patches+1, embedding_dim)
        x = x[:, 0, :]  # (batch_size, embedding_dim)
        x = self.fc(x)  # (batch_size, num_classes)
        return x
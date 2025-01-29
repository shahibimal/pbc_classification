import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.cuda.amp import autocast
from functools import partial

def efficient_attention(query, key, value):
    # Efficient attention mechanism (scaled dot-product)
    attn_scores = torch.matmul(query, key.transpose(-2, -1))
    attn_scores = attn_scores / query.size(-1)**0.5
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, value)
    return attn_output

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

        # Initialize weights manually (brute force initialization)
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize convolutional layer using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return self.norm(x)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)

        # Initialize weights manually
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize linear layers using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        
        nn.init.xavier_uniform_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn_out = efficient_attention(query, key, value)  # Efficient attention
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.out(attn_out)
        return x + self.dropout(attn_out)

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        # Use Xavier initialization 
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        # Initialize biases to 0 
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout)

        self.norm2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)

        # Initialize weights manually
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize LayerNorm and Attention components using Xavier or similar initialization
        pass  # LayerNorm does not need explicit initialization, as it's done by default in PyTorch

    def forward(self, x):
        # Ensure the tensor is moved to the appropriate device (GPU/CPU)
        device = x.device

        # Use autocast for mixed precision 
        with autocast():
            x = x + self.attn(self.norm1(x))
        
        with autocast():
            x = x + self.mlp(self.norm2(x))

        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=5, 
                 embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights manually
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize the CLS token and positional encoding
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        # Initialize the final head
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # Ensure the input tensor is on the correct device (GPU/CPU)
        device = x.device

        # Patch Embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add CLS token and positional encoding
        B, N, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through Transformer Encoder Blocks
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm and Classification
        x = self.norm(x)
        x = x[:, 0]  # Use CLS token
        x = self.head(x)

        return x

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViT(img_size=224, patch_size=16, num_classes=8).to(device)
print(model)

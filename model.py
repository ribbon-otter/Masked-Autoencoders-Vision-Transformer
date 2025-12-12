import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, dilation=1, img_size=224, patch_size=16, in_channels=3, embed_dim=784):
        super().__init__()
        self.patch_size = patch_size

        assert (dilation * (patch_size - 1) - 1) % 2 == 0
        self.padding = (dilation * (patch_size - 1) - 1) / 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, dilation=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class PatchEmbout(nn.Module):
    def __init__(self, img_size=224, patch_size=16, out_channels=3, embed_dim=784):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.img_size_in_patches = self.img_size // self.patch_size
        assert self.img_size % self.patch_size == 0, \
            f"{img_size} is not divisible by {patch_size}"
        self.proj = nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x : torch.Tensor):
        B, P, C = x.shape
        x = x.reshape([B, -1, self.img_size_in_patches, self.img_size_in_patches])
        x = self.proj(x)
        return x.sigmoid()

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.attn = SelfAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=784, num_heads=7, depth=3, pos_depth=1, mlp_dim=784*2, drop_fraction=.75):
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        self.patch_embedding2 = PatchEmbedding(3)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.pos_transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(pos_depth)
        ])
        self.output_maker = PatchEmbout(img_size, patch_size, 3, embed_dim)
        self.drop_fraction = drop_fraction
        self.long_token_length =  (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, self.long_token_length, embed_dim)) 

    def forward(self, i):
        B = i.size(0)
        x = self.patch_embedding(i)
        y = self.patch_embedding2(i)
        x = x + y 

        if self.training:
            B, Tokens, Features = x.shape
            
            indexes = torch.randperm(Tokens)[:int(Tokens*(1-self.drop_fraction))]
            x = x[:, indexes, :] 

        for block in self.transformer_blocks:
            x = block(x)
        
        if self.training:
            #we need to create empty tokens for making the output
            #create a new tensor of the old shape, but with zeros 
            #in the places we removed data
            a = torch.zeros([B, Tokens, Features], device=x.device, dtype=x.dtype)
            a[:,indexes, :] = x
            x = a

        x = x + self.pos_embed
        for block in self.pos_transformer_blocks:
            x = block(x)
        return self.output_maker(x)


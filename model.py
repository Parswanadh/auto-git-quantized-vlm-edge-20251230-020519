import torch
import torch.nn as nn

class Q4Linear(nn.Module):
    """4-bit quantized linear layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        w_q = torch.round(self.weight / 16).clamp(-8, 7) * 16
        return torch.nn.functional.linear(x, w_q)

class QuantizedVLM(nn.Module):
    """4-bit quantized Vision-Language Model"""
    def __init__(self, d_model=512, vocab_size=50000):
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, d_model, 16, stride=16),
            nn.Flatten(2),
        )
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.fusion = Q4Linear(d_model * 2, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, image, text):
        v = self.vision_encoder(image).mean(dim=-1)
        t = self.text_embed(text).mean(dim=1)
        return self.head(self.fusion(torch.cat([v, t], dim=-1)))

if __name__ == "__main__":
    model = QuantizedVLM()
    print(f"Parameters: {sum(p.numel() for p in model())/1e6:.1f}M")

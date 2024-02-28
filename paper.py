import torch.nn as nn
import torch

a = torch.randn(1, 2, 2, 2)
b = torch.randn(1, 3, 2, 2)
x = torch.cat([a, b], 1)

print(x.shape)

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(4*self.channels, self.channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.conv(x)

down = SPDConv(5)
x = down(x)

print(x.shape)
from seagul.nn import MLP
import torch

base = MLP(4,32,2,32)
b1 = MLP(16,2,16,2)
b2 = MLP(16,2,16,2)

x0 = torch.randn(1,4)
x1 = base(x0)

xl = b1(x1[..., :16])
xr = b2(x1[..., 16:])
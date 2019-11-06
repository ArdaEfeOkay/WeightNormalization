import torch
import torch.nn as nn
from MyWeightNorm import weight_norm

torch.manual_seed(0)
a = torch.randn(2, 3, dtype=torch.double)
print(a)
print(a.size())
b = torch.norm(a, p=2, dim=0)
#torch.sum(b)
print("Print B:", b)

conv1 = nn.Conv1d(2,3,1)
w = conv1.weight
print(w)
print(w.size())

normalized = weight_norm(conv1)
w2 = conv1.weight
print(w2)
print(w2.size())
print(torch.norm(w2, p=2, dim=0))

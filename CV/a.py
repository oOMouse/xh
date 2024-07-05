import torch
import numpy as np

a = torch.tensor([1, 2, 3])
x, y, z = a

b = [0] * 3
b = np.array(b)
b[[0, 1, 2]] = a
print(b)

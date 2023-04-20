from cube import *
import torch
import torch.nn as nn
import torch.nn.functional as F
c0=cube()
a=torch.tensor(c0.arr.flatten().astype(np.int64))
print(a-1)
print(F.one_hot(a-1))
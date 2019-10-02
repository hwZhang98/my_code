import torch
a = torch.tensor([[1,2,3]])
b = torch.tensor([[1],[1],[1]])
a = a.expand(3,3)
a[:,0] = a[:,0] +b[:,0]
print(a)

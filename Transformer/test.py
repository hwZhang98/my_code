import torch
a = torch.tensor([1,0,1])
b = torch.ones((2,3))
c = torch.nonzero(a==1)
print(c)
c = c.squeeze()
print(b)





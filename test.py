import torch

#declare a tensor of size 1x10
a = torch.tensor([1,1,0,0,1,1,0,0,0,0,0,0])
b = torch.tensor([10,9,8,7,6,5,4,3,2,1])

print(torch.cat((a,b),0)) #concatenate a and b along the 0th dimension

print(torch.bincount(a)) #count the number of occurrences of each value in a
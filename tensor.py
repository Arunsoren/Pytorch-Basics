import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype= torch.float32, device=device, requires_grad=True) #for autograd(backprop)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Initialization Methods
x = torch.empty(size=(3, 3))
x = torch.zeros((3, 3))
x = torch.rand((3, 3))
x = torch.ones((3, 3))
x = torch.eye(5, 5)  #Identity
#print(x)
x = torch.arange(start=0, end=5, step = 1)
#print(x)
x = torch.linspace(start=0.1, end=1, steps=10)
#print(x)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
x = torch.empty(size=(1, 5)).uniform_(0, 1)  #uniform distribution
x = torch.diag(torch.ones(3))


# Iniitializr and Convert 
tensor = torch.arange(4) #default strt=0, step=1
print(tensor.bool())   #convert to bool
print(tensor.short())  #convert to int16
print(tensor.long())   #convert to int64(Important)
print(tensor.half())   #convert to float16
print(tensor.float())  #convert to float32(Important)
print(tensor.double()) #convert to float64


# Array to Tensor Conversation and vice-verse
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
import torch

batch_size= 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape) #x[0, :] #25
print(x[:, 0].shape)  #10

# for 3rd element
print(x[2, 0:10]) #0:10 --->[0, 1, 2, ..........., 9]

x[0,0] = 100 #assign

# Fancy Indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices]) #print 3rd, 6th, 9th example in the batch

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows,cols].shape) #two elements

# More advanced Indexing
x = torch.arange(10)
print(x[(x<2) | (x> 8)])
print(x[x.remainder(2) ==0]) #even number

# Useful Operations
print(torch.where(x>5, x, x*2)) #x*2 values followed by x values
print(torch.tensor([0,0,1,2,2,3,4]).unique())  #Unique Values
print(x.dimentions())  #5x5x5
print(x.numel())

















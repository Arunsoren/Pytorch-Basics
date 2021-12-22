import torch

x = torch.arange(9)

x_3x3 = x.view(3, 3)          #view is contogious memory
print(x_3x3)
x_3x3 = x.reshape(3, 3)       #reshape it doesn't matter


y = x_3x3.t() # Transpose
print(y)
print(y.view(9))  # show error as Transpose is not a Contigious block of memory
print(y.contigious().view(9)) #will work or simply use Reshape()

# Concat
x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim = 0).shape)
print(torch.cat((x1, x2), dim = 1).shape)

z = x1.view(-1)
print(z.shape)     # Flatten

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)     #64, 10 as we add batch


z = x.permute(0, 2, 1)   #to make Transpose
print(z) #[64, 5, 2]

# Unsqueeze
x = torch.aramge(10) #[10]
print(x.unsqueeze(0).shape)  #to make a [1,10] vector
print(x.unsqueeze(1).shape)  # to make a [10, 1] vector

x = torch.arange(10).unsqueeze(0).unsqueeze(1) #1x1x10

z = x.squeeze(1) #1x10
print(z)














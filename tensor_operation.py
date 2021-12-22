import torch 

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)

z2 = torch.add(x, y)
z = x + y

#Subtraction
z = x-y

#Division
z = torch.true_divide(x, y) #element wise division

#inplace operation
t = torch.zeros(3)
t.add_(x)
t += x

#exponentiation
z = x.pow(2)
z = x ** 2

#simple comparission
z = x > 0
z = x < 0


# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm((x1, x2)) #2x3
x3 = x1.mm(x2)

# Matrix Exonentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

#Element wise Multi
z = x * y
print(z)

#dot product
z = torch.dor(x, y)

# Batch Matrix Multiplication
batch= 32
n = 10
m = 28
p = 30

tensor1 = torch.rand((batch, n, m)) #additional dim for batch
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)  # (batch, n, p)

# Example of Broadcasting

x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1-x2
z = x1 ** x2


# Other operation
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim = 0)
values, indices = torch.min(x, dim = 0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim =0)
z = torch.argmin(x, dim =0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y) #check for equal

sorted_y, indices = torch.sort(y, dim =0, descending= False)

z = torch.clamp(x, min=0)

x = torch.tensor([1,0,1,1,1], dtype= torch.bool)
z = torc.any(x)     #any true value then true
print(x)
z = torch.all(x)  #all true value then true





















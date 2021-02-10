from tinygrad.tensor import Tensor
import numpy as np

x = Tensor.eye(3).gpu()
y = Tensor([[2.0, 0, -2.0]]).gpu()
z = y.matmul(x)
print(z)
z = z.sum()
print(z)
z.backward()
print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
print()
x1 = Tensor.eye(3)
y1 = Tensor([[2.0, 0, -2.0]])
z1 = y1.matmul(x1)
print(z1)
z1 = z1.sum()
print(z1)
z1.backward()
print(x1.grad)  # dz/dx
print(y1.grad)  # dz/dy
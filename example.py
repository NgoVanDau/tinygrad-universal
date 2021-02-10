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

x1 = np.eye(3)
y1 = np.array([[2.0, 0, -2.0]])
print("np: ", np.matmul(y1, x1))

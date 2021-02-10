from tinygrad.tensor import Tensor

x = Tensor.eye(3).gpu()
y = Tensor([[2.0, 0, -2.0]]).gpu()
z = y.matmul(x).sum()
print(z)
z.backward()

print(x.grad)  # dz/dx
print(y.grad)  # dz/dy

import numpy as np
import reikna.cluda as cluda
from reikna.linalg import MatrixMul

api = cluda.cuda_api()
thr = api.Thread.create()

a = np.array([[1, 0],
              [0, 1]])
b = np.array([[4, 1],
              [2, 2]])
a_dev = thr.to_device(a)
b_dev = thr.to_device(b)

res_dev = thr.array((a.shape[0], b.shape[1]), dtype=np.float32)

dot = MatrixMul(a_dev, b_dev, out_arr=res_dev)
dotc = dot.compile(thr)
dotc(res_dev, a_dev, b_dev)
print(res_dev.get())
import numpy as np
import reikna.cluda as cluda
from reikna.algorithms import Reduce
from reikna.linalg import MatrixMul
from reikna.algorithms.predicates import predicate_sum

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

s = np.array([[0, 1], [0, 5]])
a_dev = thr.to_device(s)
print(np.sum(s, axis=0))
print(np.sum(s, axis=1))

axis = 1
rd = Reduce(a, predicate_sum(s.dtype), axes=(axis,) if axis is not None else None)
b_dev = thr.empty_like(rd.parameter.output)
rdc = rd.compile(thr)
rdc(b_dev, a_dev)
print(b_dev.get())

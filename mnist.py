import os
from tqdm import trange
import numpy as np
from tinygrad.tensor import Tensor
import tinygrad.optim as optim
from utils import fetch, get_parameters


class TinyConvNet:
    def __init__(self):
        conv = 3
        inter_chan, out_chan = 8, 16  # for speed
        self.c1 = Tensor.uniform(inter_chan, 1, conv, conv)
        self.c2 = Tensor.uniform(out_chan, inter_chan, conv, conv)
        self.l1 = Tensor.uniform(out_chan * 5 * 5, 10)

    def parameters(self):
        return get_parameters(self)

    def forward(self, x):
        x = x.reshape(shape=(-1, 1, 28, 28))  # hacks
        x = x.conv2d(self.c1).relu().max_pool2d()
        x = x.conv2d(self.c2).relu().max_pool2d()
        x = x.reshape(shape=[x.shape[0], -1])
        return x.dot(self.l1).logsoftmax()


def fetch_mnist():
    import gzip
    parse = lambda dat: np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
    return (parse(fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))[0x10:].reshape(
        (-1, 28 * 28)).astype(np.float32),
            parse(fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))[8:],
            parse(fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"))[0x10:].reshape(
        (-1, 28 * 28)).astype(np.float32),
            parse(fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"))[8:])


def sparse_categorical_crossentropy(out, Y):
    num_classes = out.shape[-1]
    YY = Y.flatten()
    y = np.zeros((YY.shape[0], num_classes), np.float32)
    y[range(y.shape[0]), YY] = -1.0 * num_classes
    y = y.reshape(list(Y.shape) + [num_classes])
    y = Tensor(y)
    return out.mul(y).mean()


X_train, Y_train, X_test, Y_test = fetch_mnist()
np.random.seed(1337)
model = TinyConvNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

Tensor.training = True
losses, accuracies = [], []
for _ in (t := trange(200, disable=os.getenv('CI') is not None)):
    samp = np.random.randint(0, X_train.shape[0], size=128)
    x = Tensor(X_train[samp])
    y = Y_train[samp]
    out = model.forward(x)
    loss = sparse_categorical_crossentropy(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cat = np.argmax(out.cpu().data, axis=-1)
    accuracy = (cat == y).mean()
    loss = loss.cpu().data
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

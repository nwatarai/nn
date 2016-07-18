import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer.functions.activation.softmax import softmax
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class SimpleCNN(Chain):

    def __init__(self, input_channel, output_channel, filter_height, filter_width, mid_units, n_units, n_label, n_kernel):       
        super(SimpleCNN, self).__init__(
            conv1 = L.Convolution2D(input_channel, output_channel, (filter_height, filter_width)),
            l1    = L.Linear(mid_units, n_units),
            l2    = L.Linear(n_units,  n_label),
        )
        self.n_kernel = n_kernel

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), self.n_kernel)
        h2 = F.dropout(F.relu(self.l1(h1)))
        y = self.l2(h2)
        return softmax(y)

    def predict(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), self.n_kernel)
        h2 = F.dropout(F.relu(self.l1(h1)))
        y = self.l2(h2)
        return softmax(y)

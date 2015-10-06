import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
import random

class TestTranspose(unittest.TestCase):

    def setUp(self):
        self.data = []
        for ndim in range(2, 5):
            shape = range(1, ndim + 1)
            random.shuffle(shape)
            axes = range(0, ndim)
            random.shuffle(axes)
            y_shape = [shape[axis] for axis in axes]
            x = numpy.random.uniform(-1, 1, tuple(shape)).astype(numpy.float32)
            gy = numpy.random.uniform(-1, 1, tuple(y_shape)).astype(numpy.float32)
            self.data.append((x, gy, axes))

        shape = range(1, 5)
        random.shuffle(shape)
        y_shape = shape[::-1]
        x = numpy.random.uniform(-1, 1, tuple(shape)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, tuple(y_shape)).astype(numpy.float32)
        self.data.append((x, gy, None))

    def check_forward(self, x_data, axes):
        x = chainer.Variable(x_data)
        y = functions.transpose(x, axes)
        self.assertEqual(y.data.dtype, numpy.float32)
        if axes is None:
            self.assertTrue(cuda.to_cpu(x_data.transpose() == y.data).all())
        else:
            self.assertTrue(cuda.to_cpu(x_data.transpose(axes) == y.data).all())

    def test_forward_cpu(self):
        for x, gy, axes in self.data:
            self.check_forward(x, axes)

    @attr.gpu
    def test_forward_gpu(self):
        for x, gy, axes in self.data:
            self.check_forward(cuda.to_gpu(x), axes)

    def check_backward(self, x_data, y_grad, axes):
        x = chainer.Variable(x_data)
        y = functions.transpose(x, axes)
        y.grad = y_grad
        y.backward()

        if axes is None:
            self.assertTrue(cuda.to_cpu(y_grad.transpose() == x.grad).all())
        else:
            ia = [0] * len(axes)
            for i, axis in enumerate(axes):
                ia[axis] = i
            self.assertTrue(cuda.to_cpu(y_grad.transpose(ia) == x.grad).all())

    def test_backward_cpu(self):
        for x, gy, axes in self.data:
            self.check_backward(x, gy, axes)

    @attr.gpu
    def test_backward_gpu(self):
        for x, gy, axes in self.data:
            self.check_backward(cuda.to_gpu(x), cuda.to_gpu(gy), axes)


testing.run_module(__name__, __file__)

from chainer import function
from chainer.utils import type_check

class Transpose(function.Function):
    """Permute the dimensions of an input array."""

    def __init__(self, axes):
        self.axes = axes
        if self.axes is not None:
            self.inv_axes = [0] * len(self.axes)
            for i, axis in enumerate(self.axes):
                self.inv_axes[axis] = i

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
        )

    def forward(self, x):
        if self.axes is None:
            return x[0].transpose(),
        return x[0].transpose(self.axes),

    def backward(self, x, gy):
        if self.axes is None:
            return gy[0].transpose(),
        return gy[0].transpose(self.inv_axes),

def transpose(x, axes=None):
    """Permute the dimensions of an input array.

    Args:
        x (~chainer.Variable): Input variable.
        axes (list of ints): By default, reverse the dimensions,
        otherwise permute the axes according to the values given.

    Returns:
        ~chainer.Variable: Variable that holds a permuted version of the input variable.

    """
    return Transpose(axes)(x)

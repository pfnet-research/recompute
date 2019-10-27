# Copied from Chainer's code base (unused in our experiments)
import chainer
import chainer.links as L

import numpy
import six

from chainer.backends import cuda
from chainer.functions.array import split_axis
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable

from chainer.backends import intel64
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


def _extract_gates(x):
    r = x.reshape((len(x), x.shape[1] // 4, 4) + x.shape[2:])
    return [r[:, :, i] for i in six.moves.range(4)]


def _sigmoid(x, xp=numpy):
    half = x.dtype.type(0.5)
    return xp.tanh(x * half) * half + half


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_grad_sigmoid(x):
    return x * (1 - x) * (1 - 2 * x)


def _grad_tanh(x):
    return 1 - x * x


def _grad_grad_tanh(x, gx):
    return -2 * x * gx


_preamble = '''
template <typename T> __device__ T sigmoid(T x) {
    const T half = 0.5;
    return tanh(x * half) * half + half;
}
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

#define COMMON_ROUTINE \
    T aa = tanh(a); \
    T ai = sigmoid(i_); \
    T af = sigmoid(f); \
    T ao = sigmoid(o);
'''


class LSTMFunction(function_node.FunctionNode):

    """Long short-term memory unit with forget gate.

    It has two inputs (c, x) and two outputs (c, h), where c indicates the cell
    state. x must have four times channels compared to the number of units.

    """

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('c', 'x'))
        c_type, x_type = in_types

        type_check.expect(
            c_type.dtype.kind == 'f',
            x_type.dtype == c_type.dtype,

            c_type.ndim >= 2,
            x_type.ndim >= 2,
            c_type.ndim == x_type.ndim,

            x_type.shape[0] <= c_type.shape[0],
            x_type.shape[1] == 4 * c_type.shape[1],
        )
        for i in six.moves.range(2, type_check.eval(c_type.ndim)):
            type_check.expect(x_type.shape[i] == c_type.shape[i])

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        c_prev, x = inputs
        a, i, f, o = _extract_gates(x)
        batch = len(x)

        if isinstance(x, chainer.get_cpu_array_types()):
            if intel64.should_use_ideep('>=auto'):
                xp = intel64.ideep.get_array_module(x)
            else:
                xp = numpy
            a = xp.tanh(a)
            i = _sigmoid(i, xp)
            f = _sigmoid(f, xp)
            o = _sigmoid(o, xp)

            c_next = numpy.empty_like(c_prev)
            c_next[:batch] = a * i + f * c_prev[:batch]
            h = o * xp.tanh(c_next[:batch])
        else:
            c_next = cuda.cupy.empty_like(c_prev)
            h = cuda.cupy.empty_like(c_next[:batch])
            cuda.elementwise(
                'T c_prev, T a, T i_, T f, T o', 'T c, T h',
                '''
                    COMMON_ROUTINE;
                    c = aa * ai + af * c_prev;
                    h = ao * tanh(c);
                ''',
                'lstm_fwd', preamble=_preamble)(
                    c_prev[:batch], a, i, f, o, c_next[:batch], h)

        c_next[batch:] = c_prev[batch:]
        self.retain_outputs((0,))
        return c_next,

    def backward(self, indexes, grads):
        grad_inputs = (
            self.get_retained_inputs() + self.get_retained_outputs() + grads)
        return chainer.functions.activation.lstm.LSTMGrad()(*grad_inputs)


def lstm(c_prev, x):
    return LSTMFunction().apply((c_prev, x))


class LSTMBase(link.Chain):

    def __init__(self, in_size, out_size=None, lateral_init=None,
                 upward_init=None, bias_init=None, forget_bias_init=None):
        if out_size is None:
            out_size, in_size = in_size, None

        super(LSTMBase, self).__init__()
        if bias_init is None:
            bias_init = 0
        if forget_bias_init is None:
            forget_bias_init = 1
        self.state_size = out_size
        self.lateral_init = lateral_init
        self.upward_init = upward_init
        self.bias_init = bias_init
        self.forget_bias_init = forget_bias_init

        with self.init_scope():
            self.upward = linear.Linear(in_size, 4 * out_size, initialW=0)
            self.lateral = linear.Linear(out_size, 4 * out_size, initialW=0,
                                         nobias=True)
            if in_size is not None:
                self._initialize_params()

    def _initialize_params(self):
        lateral_init = initializers._get_initializer(self.lateral_init)
        upward_init = initializers._get_initializer(self.upward_init)
        bias_init = initializers._get_initializer(self.bias_init)
        forget_bias_init = initializers._get_initializer(self.forget_bias_init)

        for i in six.moves.range(0, 4 * self.state_size, self.state_size):
            lateral_init(self.lateral.W.array[i:i + self.state_size, :])
            upward_init(self.upward.W.array[i:i + self.state_size, :])

        a, i, f, o = chainer.functions.activation.lstm._extract_gates(
            self.upward.b.array.reshape(1, 4 * self.state_size, 1))

        bias_init(a)
        bias_init(i)
        forget_bias_init(f)
        bias_init(o)


class LSTM(LSTMBase):

    def __init__(self, in_size, out_size=None, lateral_init=None,
                 upward_init=None, bias_init=None, forget_bias_init=None):
        if out_size is None:
            in_size, out_size = None, in_size
        super(LSTM, self).__init__(
            in_size, out_size, lateral_init, upward_init, bias_init,
            forget_bias_init)
        self.reset_state()

    def _to_device(self, device, skip_between_cupy_devices=False):
        # Overrides Link._to_device
        # TODO(niboshi): Avoid forcing concrete links to override _to_device
        device = chainer.get_device(device)
        super(LSTM, self)._to_device(
            device, skip_between_cupy_devices=skip_between_cupy_devices)
        if self.c is not None:
            if not (skip_between_cupy_devices
                    and device.xp is cuda.cupy
                    and isinstance(self.c, cuda.ndarray)):
                self.c.to_device(device)
        if self.h is not None:
            if not (skip_between_cupy_devices
                    and device.xp is cuda.cupy
                    and isinstance(self.h, cuda.ndarray)):
                self.h.to_device(device)
        return self

    def set_state(self, c, h):
        """Sets the internal state.

        It sets the :attr:`c` and :attr:`h` attributes.

        Args:
            c (~chainer.Variable): A new cell states of LSTM units.
            h (~chainer.Variable): A new output at the previous time step.

        """
        assert isinstance(c, variable.Variable)
        assert isinstance(h, variable.Variable)
        c.to_device(self.device)
        h.to_device(self.device)
        self.c = c
        self.h = h

    def reset_state(self):
        """Resets the internal state.

        It sets ``None`` to the :attr:`c` and :attr:`h` attributes.

        """
        self.c = self.h = None

    def forward(self, x):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        if self.upward.W.array is None:
            with chainer.using_device(self.device):
                in_size = utils.size_of_shape(x.shape[1:])
                self.upward._initialize_params(in_size)
                self._initialize_params()

        batch = x.shape[0]
        lstm_in = x
        # lstm_in = self.upward(x)
        h_rest = None
        if self.h is not None:
            h_size = self.h.shape[0]
            if batch == 0:
                h_rest = self.h
            elif h_size < batch:
                msg = ('The batch size of x must be equal to or less than'
                       'the size of the previous state h.')
                raise TypeError(msg)
            elif h_size > batch:
                h_update, h_rest = split_axis.split_axis(
                    self.h, [batch], axis=0)
                lstm_in += self.lateral(h_update)
            else:
                lstm_in += self.lateral(self.h)
        if self.c is None:
            with chainer.using_device(self.device):
                self.c = variable.Variable(
                    self.xp.zeros((batch, self.state_size), dtype=x.dtype))
        self.c, = lstm(self.c, lstm_in)

        return self.c


class LongLSTM(chainer.Chain):
    def __init__(self):
        super().__init__()

        with self.init_scope():
            self.layer = LSTM(200, 50)

    def __call__(self, *args):
        print(args)
        print(len(args))
        xs, c0, t = args[:-2], args[-2], args[-1]
        c0 = chainer.as_variable(c0)
        self.layer.set_state(c0, c0)
        for i in range(len(xs)):
            h = self.layer(xs[i])
        h = chainer.functions.mean_absolute_error(h, t)
        return h


class LongLinear(chainer.Chain):
    def __init__(self):
        super().__init__()

        with self.init_scope():
            self.lin = L.Linear(1000, 1000)

    def __call__(self, x):
        h = x
        for _ in range(256):
            h = self.lin(h)

        return h

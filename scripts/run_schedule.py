import argparse
from collections import defaultdict
from collections import namedtuple
import json
import random
import time
import weakref

import chainer
from chainer.functions.connection.linear import LinearFunction  # noqa
from chainer.functions.activation.relu import ReLU  # noqa
from chainer.functions.loss.softmax_cross_entropy import SoftmaxCrossEntropy  # noqa
from chainer.functions.math.basic_math import Add  # noqa
from chainer.functions.math.basic_math import MulConstant  # noqa
from chainer.functions.connection.convolution_2d import Convolution2DFunction  # noqa
from chainer.functions.normalization.batch_normalization import BatchNormalization  # noqa
from chainer.functions.pooling.average_pooling_2d import AveragePooling2D  # noqa
from chainer.functions.pooling.max_pooling_2d import MaxPooling2D  # noqa
from chainer.functions.normalization.local_response_normalization import LocalResponseNormalization  # noqa
from chainer.functions.pooling.upsampling_2d import Upsampling2D  # noqa
from chainer.functions.array.reshape import Reshape  # noqa
from chainer.functions.array.concat import Concat  # noqa
from chainer.functions.noise.dropout import Dropout  # noqa
from chainer.functions.connection.deconvolution_2d import Deconvolution2DFunction  # noqa
from chainer.functions.loss.mean_absolute_error import MeanAbsoluteError  # noqa
from chainer.functions.array.resize_images import ResizeImages  # noqa
from chainer.functions.array.get_item import GetItem  # noqa
# from chainer.functions. import   # noqa
from long_rnn import LSTMFunction  # noqa
import numpy as np

import cupy
from cupy.cuda import memory_hook

from zoo import zoo_list


FunctionNodeInfo = namedtuple('FunctionNodeInfo',
                              ('inputs', 'outputs', 'name', 'mode', 'args'))


class PeakProfileHook(memory_hook.MemoryHook):

    def __init__(self):
        self.peak = 0

    def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id):
        self.peak = max(self.peak,
                        chainer.backends.cuda.memory_pool.used_bytes())


def single_backward(fn, indices, retained_inputs, o_grads,
                    i_grads,
                    outputs=None):
    input_vars = [chainer.as_variable(z) for z
                  in retained_inputs]
    fn.inputs = tuple([z.node for z in input_vars])
    if outputs is not None:
        output_vars = [chainer.as_variable(z) for z
                       in outputs]
        fn.outputs = tuple([weakref.ref(z.node) for z in output_vars])
        fn._retained_output_data = tuple(outputs)

    i_grads = fn.backward_accumulate(indices, o_grads, i_grads)

    if len(i_grads) is not len(indices):
        i_grads_ = []
        for i in indices:
            i_grads_.append(i_grads[i])
        i_grads = i_grads_

    input_vars = None
    fn.inputs = None
    output_vars = None
    fn.outputs = None
    fn._retained_output_data = None

    def unwrap(v):
        if isinstance(v, chainer.Variable):
            return v.data
        else:
            return v

    return (unwrap(v) for v in i_grads)


def get(f):
    return f.readline().rstrip().split()


def function_initialization(name, args, batchsize):
    if name == 'LinearFunction':
        return ''
    elif name == 'ReLU':
        return ''
    elif name == 'SoftmaxCrossEntropy':
        return ''
    elif name == 'Add':
        return ''
    elif name == 'Convolution2DFunction':
        return 'stride={}, pad={}, cover_all={}, dilate={}'.format(
            args['sy'], args['ph'], args['cover_all'],
            args['dy'])
    elif name == 'BatchNormalization':
        return 'eps={}, decay={}, axis={}'.format(
            args['eps'], args['decay'], tuple(args['axis']))
    elif name == 'Softmax':
        return ''
    elif name == 'Reshape':
        # hardcode batchsize
        return '{}'.format((batchsize,) + tuple(args['shape'][1:]))
    elif name == 'AveragePooling2D':
        return '{}, stride={}, pad={}, cover_all={}'.format(
            args['kh'], args['sy'], args['ph'],
            args['cover_all'])
    elif name == 'MaxPooling2D':
        return '{}, stride={}, pad={}, cover_all={}'.format(
            args['kh'], args['sy'], args['ph'],
            args['cover_all'])
    elif name == 'LocalResponseNormalization':
        return ''
    elif name == 'Upsampling2D':
        # todo: indices are required
        return ''
    elif name == 'Concat':
        return 'axis={}'.format(args['axis'])
    elif name == 'Dropout':
        # print('\033[91mDropout is used. Gradient check will fail.\033[0m:')
        return 'dropout_ratio=0.0'
    elif name == 'Deconvolution2DFunction':
        return 'stride={}, pad={}, outsize=({}, {})'.format(
            args['sy'], args['ph'], args['outh'], args['outw'])
    elif name == 'MeanAbsoluteError':
        return ''
    elif name == 'LSTMFunction':
        return ''
    elif name == 'ResizeImages':
        return '({}, {})'.format(args['out_H'], args['out_W'])
    elif name == 'MulConstant':
        return '{}'.format(args['value'])
    elif name == 'GetItem':
        return '{}'.format(args['slices'])
    else:
        raise NotImplementedError(name)


def to_codename(var_id, var_mode, paramnames):
    assert var_mode == 'data' or var_mode == 'grad'
    paramname = paramnames[var_id]

    if paramname == '_':
        if var_mode == 'data':
            ret = 'h' + str(var_id)
        else:
            ret = 'gh' + str(var_id)
    elif paramname.isdigit():
        if var_mode == 'data':
            ret = 'input' + paramname
        else:
            ret = 'ginput' + paramname
    else:
        es = paramname.split('/')[1:]
        es = [('[' + e + ']') if e.isdigit() else ('.' + e) for e in es]
        ret = 'self' + ''.join(es) + '.' + var_mode

    return ret


def generate_code(cgtxt, sch, batchsize):
    # read computation graph file
    f = open(cgtxt, 'r')

    N, M = map(int, get(f))

    memories = dict()
    paramnames = dict()
    for _ in range(N):
        var_id, memory, paramname = get(f)
        paramnames[int(var_id)] = paramname
        memories[int(var_id)] = int(memory)

    fninfos = []
    for _ in range(M):
        mode, = get(f)

        k, p = map(int, get(f))

        inputs = []
        for _ in range(k):
            var_id, in_type = get(f)
            inputs.append((int(var_id), in_type))

        outputs = []
        for _ in range(p):
            var_id, = get(f)
            outputs.append(int(var_id))

        fun_name = f.readline().rstrip()  # do not split
        args = json.loads(f.readline().rstrip())

        fninfos.append(FunctionNodeInfo(inputs, outputs, fun_name,
                                        mode, args))

    f.close()

    # determin terminals
    terminals = set(range(N))
    for i, fn in enumerate(fninfos):
        if i >= M // 2:
            break
        for var_id, _ in fn.inputs:
            terminals.discard(var_id)

    # read schedule file
    f = open(sch, 'r')
    code = []
    alive_grads = set()
    for line in f:
        chunk = line.rstrip().split()
        op, idx = chunk[:2]
        idx = int(idx)

        if op == 'compute':
            fn = fninfos[idx]

        if op == 'compute' and fn.mode == 'forward':
            init_args = function_initialization(fn.name, fn.args, batchsize)
            code.append('fn{} = {}({})'.format(idx, fn.name, init_args))
            input_args = ', '.join(
                [to_codename(var_id, 'data', paramnames)
                 for var_id, in_mode in fn.inputs]) + ','
            output_args = ', '.join(
                [to_codename(var_id, 'data', paramnames)
                 for var_id in fn.outputs]) + ','
            code.append('{} = fn{}.forward(({}))'.format(
                output_args, idx, input_args))

            out_var = fn.outputs[0]
            if out_var in terminals:
                v = to_codename(out_var, 'data', paramnames)
                g = to_codename(out_var, 'grad', paramnames)
                code.append('{} = chainer.as_variable(self.xp.ones_like({}))'
                            .format(g, v))
                alive_grads.add(out_var)

            '''
            if fn.name == 'ReLU':
                # special rule: relu input is not required in backward part
                v = to_codename(fn.inputs[0][0], 'data', paramnames)
                code.append('{} = None'.format(v))
            '''

        elif op == 'compute' and fn.mode == 'backward':
            indices = tuple(map(int, chunk[2:]))
            forward_fn_idx = idx - M // 2

            inputs = defaultdict(list)
            for var_id, in_type in fn.inputs:
                if in_type == 'input' or in_type == 'output':
                    mode = 'data'
                elif in_type == 'gradient':
                    mode = 'grad'
                else:
                    raise NotImplementedError(in_type)
                inputs[in_type].append(
                    to_codename(var_id, mode, paramnames))

            forward_input_args = ', '.join(inputs['input'])
            backward_grad_args = ', '.join(inputs['gradient'])
            if len(inputs['output']) == 0:
                forward_output_args = 'None'
            else:
                forward_output_args = 'tuple([{}])'.format(
                    ', '.join(inputs['output']))

            # substitute None for new variables
            for j in indices:
                if fn.outputs[j] not in alive_grads:
                    code.append('{} = None'.format(
                        to_codename(fn.outputs[j], 'grad', paramnames)))
            return_args = ', '.join(
                [to_codename(fn.outputs[j], 'grad', paramnames)
                 for j in indices]) + ','

            code.append(('{} = single_backward(fn{}, {}, tuple([{}]), ' +
                         'tuple([{}]), tuple([{}]), outputs={})').format(
                             return_args,
                             forward_fn_idx,
                             indices,
                             forward_input_args,
                             backward_grad_args,
                             return_args,
                             forward_output_args))
            # code.append('pass; del fn{}'.format(forward_fn_idx))

            # update alive_grads
            for j in indices:
                var_id = fn.outputs[j]
                alive_grads.add(var_id)

        elif op == 'forget':
            mode = chunk[2]

            if mode == 'forward':
                v = to_codename(idx, 'data', paramnames)
            elif mode == 'backward':
                assert idx in alive_grads
                v = to_codename(idx, 'grad', paramnames)
                alive_grads.remove(idx)
            else:
                NotImplementedError(mode)

            code.append('del {}'.format(v))
        else:
            raise NotImplementedError

    return code


def customized_forward(self, code, inputs, print_code=True):
    for i, x in enumerate(inputs):
        exec('input{} = x'.format(i))
    peak_mem = 0
    with chainer.using_config('enable_backprop', False):
        for line in code:
            if print_code:
                print(line)

            peak_mem = max(peak_mem,
                           chainer.backends.cuda.memory_pool.used_bytes())

            if print_code and line.startswith('del '):
                prev_ub = chainer.backends.cuda.memory_pool.used_bytes()
                exec('varsize = {}.size * {}.dtype.itemsize'.format(
                    line[4:], line[4:]))

            exec(line)

            if print_code and line.startswith('del '):
                ub = chainer.backends.cuda.memory_pool.used_bytes()
                if prev_ub <= ub:
                    print('\033[92mWarning: Used bytes was not decreased after "{}".\033[0m'.format(line))  # noqa
                    print('\033[92mPrev: {} MB\033[0m'.format(prev_ub // 1000000))  # noqa
                    print('\033[92mCurr: {} MB\033[0m'.format(ub // 1000000))  # noqa
                    exec("print('\033[92mExpected Decrease: {} MB\033[0m'.format(varsize // 1000000))")  # noqa

    return peak_mem


def reset_seed(args):
    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    if args.gpu >= 0:
        chainer.backends.cuda.cupy.random.seed(seed)


def reset_model(args, funcall):
    reset_seed(args)
    method, batchsize = zoo_list[args.model]

    if args.batchsize is not None:
        batchsize = args.batchsize
    inputs, model = method(batchsize)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if funcall:
        small_inputs, _ = method(1)
        small_inputs = [chainer.as_variable(model.xp.array(x.data))
                        for x in small_inputs]
        model(*small_inputs)

    return inputs, model


def gradcheck(args):
    chainer.global_config.cudnn_deterministic = True

    code = generate_code(args.cg, args.sch, args.batchsize)

    inputs, model = reset_model(args, funcall=True)
    xp = model.xp
    inputs = tuple(xp.array(x.data) for x in inputs)

    # normal backward
    print('# Running normal backward')
    model.cleargrads()
    # with chainer.using_config('in_recomputing', True):
    y = model(*inputs)
    y.backward()
    grad_expected = dict()
    for name, v in model.namedparams():
        g = v.grad.copy()
        if args.gpu >= 0:
            g = chainer.cuda.to_cpu(g)
        grad_expected[name] = g
    del y

    # customized backward
    print('# Running customized backward')
    _, model = reset_model(args, funcall=True)
    model.cleargrads()
    customized_forward(model, code, inputs)
    error = False
    for name, v in model.namedparams():
        expected = grad_expected[name]
        received = v.grad.copy()
        if args.gpu >= 0:
            received = chainer.cuda.to_cpu(received)
        if not np.allclose(received, expected, atol=1e-4, rtol=1e-4):
            error = True
            print('\033[91mGradient was inconsistent\033[0m:', name)
            print('norm: ', np.linalg.norm(expected - received))
            np.set_printoptions(threshold=np.nan)
            np.set_printoptions(formatter={'float_kind': lambda x: "%.5f" % x})
    if not error:
        print('\033[92mGradient check OK!\033[0m')


def sync():
    event = cupy.cuda.stream.Event()
    event.record()
    event.synchronize()


def normal_backward(model, inputs):
    y = model(*inputs)
    y.backward()


def benchmark(args):
    inputs, model = reset_model(args, funcall=True)
    xp = model.xp
    inputs = tuple(xp.array(x.data) for x in inputs)

    model.cleargrads()
    if args.verbose:
        ub = chainer.backends.cuda.memory_pool.used_bytes()
        print('Used Bytes (Params Only)    :', ub // 1000000, 'MB')

    elapsed_ts = []

    if args.sch is None:
        for i in range(5):
            sync()
            start_t = time.time()
            model.cleargrads()

            hook = PeakProfileHook()
            with hook:
                normal_backward(model, inputs)
                sync()

            if not args.repeat:
                print('Peak Bytes: ', hook.peak // 1000000, 'MB')

            elapsed_t = time.time() - start_t

            if args.verbose:
                print('Backward Time: {:.3f}s'.format(elapsed_t))

            if i > 0:
                elapsed_ts.append(elapsed_t)

        peak = None
    else:
        code = generate_code(args.cg, args.sch, args.batchsize)

        repeat = 5 if args.repeat else 1

        for i in range(repeat):

            sync()
            start_t = time.time()
            model.cleargrads()
            hook = PeakProfileHook()
            with hook:
                customized_forward(model, code, inputs,
                                   print_code=args.verbose)
                sync()

            if not args.repeat:
                print('Peak Bytes: ', hook.peak // 1000000, 'MB')

            elapsed_t = time.time() - start_t

            if args.verbose:
                print('Backward Time: {:.3f}s'.format(elapsed_t))

            if i > 0:
                elapsed_ts.append(elapsed_t)

    if args.verbose:
        ub = chainer.backends.cuda.memory_pool.used_bytes()
        print('Used Bytes (After Backward) :', ub // 1000000, 'MB')
        if peak is not None:
            print('Used Bytes (Peak Time)      :', peak // 1000000, 'MB')
        print('Average Backward Time: {:.3f}s'.format(
            sum(elapsed_ts) / len(elapsed_ts)))
    else:
        if elapsed_ts:
            print('{:.3f}'.format(sum(elapsed_ts) / len(elapsed_ts)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('execmode', type=str,
                        help='gradcheck or benchmark')
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--cg', type=str, default='mlp.txt')
    parser.add_argument('--sch', type=str, default=None)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=None)

    parser.add_argument('--no-repeat', action='store_false', dest='repeat')
    parser.set_defaults(repeat=True)
    args = parser.parse_args()

    if args.execmode == 'gradcheck':
        gradcheck(args)
    elif args.execmode == 'benchmark':
        benchmark(args)
    else:
        raise NotImplementedError(args.execmode)


if __name__ == '__main__':
    main()

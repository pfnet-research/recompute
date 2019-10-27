# dump cg.txt for specified model

from collections import defaultdict
import heapq
import json
import sys

import chainer
from chainer import function_node
from chainer import variable

import chainer.computational_graph as c
import cupy
import numpy as np

from zoo import zoo_list


def is_primitive(t):
    if t is None:
        return True
    elif isinstance(t, int) or isinstance(t, float) or\
            isinstance(t, bool) or isinstance(t, str) or\
            isinstance(t, np.int64) or isinstance(t, np.int32) or\
            isinstance(t, cupy.int64) or isinstance(t, cupy.int32) or\
            isinstance(t, slice):
        return True
    elif isinstance(t, tuple) or isinstance(t, list):
        for x in t:
            if not is_primitive(x):
                return False
        return True
    else:
        return False


def _convert(t):
    if isinstance(t, np.int64) or isinstance(t, np.int32) or\
       isinstance(t, cupy.int64) or isinstance(t, cupy.int32):
        return int(t)
    elif isinstance(t, tuple) and any(isinstance(e, slice) for e in t):
        return str(t)
    else:
        return t


def get_bytes(t):
    return int(np.prod(t.shape) * t.dtype.itemsize)


def get_cid(raw_id, id_list):
    if raw_id in id_list:
        return id_list[raw_id]
    else:
        compact_id = len(id_list)
        id_list[raw_id] = compact_id
        return compact_id


def build_graph(outputs):
    cands = []
    seen_edges = set()
    nodes = set()
    push_count = [0]

    def add_cand(cand):
        heapq.heappush(cands, (-cand.rank, push_count[0], cand))
        push_count[0] += 1

    for o in outputs:
        if isinstance(o, variable.Variable):
            o = o.node
        add_cand(o)
        nodes.add(o)

    ins = defaultdict(list)
    outs = defaultdict(list)
    while cands:
        _, _, cand = heapq.heappop(cands)
        if isinstance(cand, variable.VariableNode):
            creator = cand.creator_node
            if creator is not None and (creator, cand) not in seen_edges:
                add_cand(creator)
                seen_edges.add((creator, cand))
                outs[id(creator)].append(id(cand))
                nodes.add(creator)
                nodes.add(cand)
        elif isinstance(cand, function_node.FunctionNode):
            for input_ in cand.inputs:
                if input_ is not cand and (input_, cand) not in seen_edges:
                    add_cand(input_)
                    seen_edges.add((input_, cand))
                    ins[id(cand)].append(id(input_))
                    nodes.add(input_)
                    nodes.add(cand)

    vs = []
    fs = []
    for node in nodes:
        if isinstance(node, variable.VariableNode):
            vs.append(node)
        elif isinstance(node, function_node.FunctionNode):
            fs.append(node)
        else:
            raise NotImplementedError(node)

    return vs, fs, ins, outs


def dump(vs, fs, ins, outs, inputs, model):
    id_list = dict()

    # number of variables and function nodes
    print('{} {}'.format(len(vs), len(fs) * 2))

    # variable info (forward)
    for v in vs:
        params = list(model.namedparams())

        name = '_'
        input_ids = [id(inpv.node) for inpv in inputs]
        if id(v) in input_ids:
            name = input_ids.index(id(v))
        for r, p in params:
            if id(v) == id(p.node):
                name = r

        print('{} {} {}'.format(
            get_cid(id(v), id_list), get_bytes(v), name))

    # function node info (forward)
    for f in fs:
        i = id(f)
        print('forward')
        print('{} {}'.format(len(ins[i]), len(outs[i])))
        print('\n'.join(str(get_cid(x, id_list)) + ' _' for x in ins[i]))
        print('\n'.join(str(get_cid(x, id_list)) for x in outs[i]))
        print(type(f).__name__)
        rs = dict((key, _convert(val)) for key, val in vars(f).items()
                  if is_primitive(val))
        print(json.dumps(rs))

    # function node info (backward)
    for f in fs:
        i = id(f)

        fn_inputs = [x.get_variable() for x in f.inputs]
        if fn_inputs is None:
            fn_inputs = []

        retained_outputs = f.get_retained_outputs()
        if retained_outputs is None:
            retained_outputs = []

        print('backward')

        backward_ins = []
        for x in outs[i]:
            backward_ins.append(str(get_cid(x, id_list)) + ' gradient')

        if type(f).__name__ != 'ReLU':
            # ReLU is exception
            for x in fn_inputs:
                backward_ins.append(
                    str(get_cid(id(x.node), id_list)) + ' input')

        for x in retained_outputs:
            backward_ins.append(str(get_cid(id(x.node), id_list)) + ' output')

        print('{} {}'.format(
            len(backward_ins),
            len(ins[i])))

        print('\n'.join(backward_ins))

        for x in ins[i]:
            print(str(get_cid(x, id_list)))

        print(type(f).__name__)
        rs = dict((key, _convert(val)) for key, val in vars(f).items()
                  if is_primitive(val))
        print(json.dumps(rs))


def main():
    method, default_batchsize = zoo_list[sys.argv[1]]
    gpu = int(sys.argv[2])

    assert gpu >= 0

    inputs, model = method(default_batchsize)
    chainer.backends.cuda.get_device_from_id(gpu).use()
    model.to_gpu()
    inputs = [chainer.as_variable(model.xp.array(x.data)) for x in inputs]
    outs = [model(*inputs)]

    g = c.build_computational_graph(outs)

    with open('a.dot', 'w') as o:
        o.write(g.dump())

    vs, fs, ins, outs = build_graph(outs)
    dump(vs, fs, ins, outs, inputs, model)


if __name__ == '__main__':
    main()

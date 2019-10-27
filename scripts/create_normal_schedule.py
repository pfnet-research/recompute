from collections import defaultdict
import sys


n, m = map(int, sys.stdin.readline().rstrip().split())

computed = set()
variable_nodes = []
for _ in range(n):
    node_id, memory, paramname = sys.stdin.readline().rstrip().split()
    node_id = int(node_id)
    memory = int(memory)
    if paramname != '_':
        # parameter
        computed.add((node_id, 'f'))

left_counts = defaultdict(int)
fns = []
for _ in range(m):
    fun_type = sys.stdin.readline().rstrip()
    p, q = map(int, sys.stdin.readline().rstrip().split())
    ins = []
    outs = []
    for _ in range(p):
        node_id, ntype = sys.stdin.readline().rstrip().split()
        node_id = int(node_id)
        if ntype == 'gradient':
            ins.append((node_id, 'b'))
        else:
            ins.append((node_id, 'f'))
    for _ in range(q):
        node_id, = sys.stdin.readline().rstrip().split()
        node_id = int(node_id)
        if fun_type == 'forward':
            node = (node_id, 'f')
        else:
            node = (node_id, 'b')
        outs.append(node)
        left_counts[node] += 1
    fun_name = sys.stdin.readline().rstrip()
    sys.stdin.readline().rstrip()  # json, unused

    fns.append((ins, outs))

terminals = set(range(n))
for i, (ins, _) in enumerate(fns):
    if i >= m // 2:
        break
    for node_id, _ in ins:
        terminals.discard(node_id)
assert(len(terminals) == 1)
for t in terminals:
    computed.add((t, 'b'))

# aho iteration
fns_used = set()

# forward is first
while len(computed) < n + 1:
    # for _ in range(2 * n):
    for i, (ins, outs) in enumerate(fns):
        if i < m // 2:
            if i in fns_used:
                continue
            if all((t in computed) for t in ins):
                # forward node
                print('compute', i)
                for t in outs:
                    left_counts[t] -= 1
                    if left_counts[t] == 0:
                        computed.add(t)
                fns_used.add(i)

# backward is second
while len(computed) < 2 * n:
    for i, (ins, outs) in enumerate(fns):
        if i >= m // 2:
            if i in fns_used:
                continue
            if all((t in computed) for t in ins):
                # backward node
                indices = ' '.join(str(i) for i in range(len(outs)))
                print('compute', i, indices)
                for t in outs:
                    left_counts[t] -= 1
                    if left_counts[t] == 0:
                        computed.add(t)
                fns_used.add(i)


'''
def name(node):
    return node[1] + str(node[0])


print('digraph G {')
print('{')
for i in range(n):
    for ch in ('f', 'b'):
        node = (i, ch)
        color = 'red' if node in computed else 'white'
        print(name(node), '[fillcolor={},style="filled"]'.format(color))
print('}')
for ins, outs in fns:
    for to in outs:
        for ti in ins:
            print(name(ti), '->', name(to))
print('}')
'''

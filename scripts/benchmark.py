import argparse
from subprocess import run, PIPE

from zoo import zoo_list


def binsearch(args):
    lo = args.blo
    hi = args.bhi + 1

    while hi - lo > 1:
        bs = (hi + lo) // 2
        print('#', bs)

        cmd = ['python3', 'run_schedule.py', 'benchmark',
               '--gpu', str(args.gpu),
               '--cg', args.cg,
               '--sch', args.sch,
               '--model', args.model,
               '--batchsize', str(bs)]

        proc = run(cmd, stdout=PIPE, stderr=PIPE)

        if 'OutOfMemoryError' in proc.stderr.decode('utf-8'):
            print('Memory Error')
            hi = bs
        else:
            print(proc.stdout.decode('utf-8').rstrip())
            lo = bs
    print('max possible batchsize: ', lo)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('method', type=str,
                        help='normal or recompute')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--cg', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--blo', type=int, required=True)
    parser.add_argument('--bhi', type=int, required=True)
    parser.add_argument('--budget', type=int, default=10 * 10 ** 9)
    parser.add_argument('--sch', type=str, default=None)
    args = parser.parse_args()

    _, default_bs = zoo_list[args.model]

    if args.sch is not None and args.method == 'recompute':
        # binary search mode
        pass
        # binsearch(args)
        # return

    for bs in range(args.blo, args.bhi + 1):
        if args.method == 'normal':
            cmd = ['python3', 'run_schedule.py', 'benchmark',
                   '--gpu', str(args.gpu),
                   '--cg', args.cg,
                   '--model', args.model,
                   '--batchsize', str(bs)]

        elif args.method == 'recompute':

            if args.sch is None:
                schfile = '/tmp/temp_sch.txt'
                # Generate sch file
                B = args.budget * default_bs // bs

                cmd = ['../dp2.out', '-b', str(B)]
                with open(args.cg, 'r') as f:
                    proc = run(cmd, stdin=f, stdout=PIPE, stderr=PIPE)

                with open('/tmp/temp_sch.txt', 'w') as f:
                    f.write(proc.stdout.decode('utf-8'))
            else:
                schfile = args.sch

            cmd = ['python3', 'run_schedule.py', 'benchmark',
                   '--gpu', str(args.gpu),
                   '--cg', args.cg,
                   '--sch', schfile,
                   '--model', args.model,
                   '--batchsize', str(bs)]

        proc = run(cmd, stdout=PIPE, stderr=PIPE)

        if 'OutOfMemoryError' in proc.stderr.decode('utf-8'):
            print('MemoryError')
            break
        else:
            print(proc.stdout.decode('utf-8').rstrip())


if __name__ == '__main__':
    main()

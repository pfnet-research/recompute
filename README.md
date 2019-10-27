# A Graph Theoretic Framework of Recomputation Algorithms for Memory-Efficient Backpropagation

This repository contains an implementation of our paper ["A Graph Theoretic Framework of Recomputation Algorithms for Memory-Efficient Backpropagation" (NeurIPS'19)](https://arxiv.org/abs/1905.11722).

## Requirements

We performed our experiments on the following environment.

- Ubuntu 16.04
- Python >= 3.6.6
- g++ >= 5.5.0

In addition, some Python packages are required. To install them, please use the following command:

```
pip install -r requirements.txt
```

and then install CuPy. Please follow the instruction in the official documentation of CuPy. https://cupy.chainer.org/

- Chainer https://github.com/chainer/chainer.git@9f243e66fc (latest master repository at the time we conducted experiments)
- Cupy==6.0.0b1
- ChainerCV https://github.com/chainer/chainercv.git@23ae16b471d59bd314e49280ca1058f5bdd7eb77

## Directories

###  `mksch`
This directory contains C++ code for obtaining a recomputation schedule of given a computational graph through dynamic programming described in the paper.
To build the code, please run `g++` from top directory.

```
# Exact solution
g++ -std=c++11 -O3 mksch/DP.cpp mksch/dag_dp.cpp -o a.out
```

or

```
# Approximate solution
g++ -std=c++11 -O3 mksch/DP2.cpp mksch/dag_dp.cpp -o a.out
```

To run the compiled binary, please specify memory budget by `-b` option. Argument of `-b` should be specified in bytes.
Example:
```
# 2G memory budget & time centric strategy
./a.out -b 2000000000 < computational_graphs/cg_resnet.txt > schedule.txt
```

Memory centric option is also available. Please specify `-t M` in command line option.
Example:
```
# 2G memory budget & memory centric strategy
./a.out -b 2000000000 -t M < computational_graphs/cg_resnet.txt > schedule.txt
```

### `computational_graphs`
This directory contains computational graph files for experiments.
These files are encoded in plain text in our intermediate representation.

### `scripts`
This directory contains python scripts for benchmarking performance of the recomputation strategy.

```
python3 run_schedule.py benchmark --cg ../computational_graphs/cg_resnet50.txt --model resnet50 --batchsize 96 --sch schedule.txt
```

Several options are available. Please see the `argparse` part in the code.

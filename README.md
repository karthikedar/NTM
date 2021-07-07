# NTM Pytorch Implementation

PyTorch implementation of [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (NTM).

An **NTM** is a memory augumented neural network (attached to external memory) where the interactions with the external memory (address, read, write) are done using differentiable transformations. Overall, the network is end-to-end differentiable and thus trainable by a gradient based optimizer.

The NTM is processing input in sequences, much like an LSTM, but with additional benfits: (1) The external memory allows the network to learn algorithmic tasks easier (2) Having larger capacity, without increasing the network's trainable parameters.

The external memory allows the NTM to learn algorithmic tasks, that are much harder for LSTM to learn, and to maintain an internal state much longer than traditional LSTMs.

## A PyTorch Implementation

This repository implements a vanilla NTM in a straight forward way. The following architecture is used:

![NTM Architecture](./images/ntm.png)

### Features
* Batch learning support
* Numerically stable
* Flexible head configuration - use X read heads and Y write heads and specify the order of operation
* **copy** and **repeat-copy** experiments agree with the paper

***

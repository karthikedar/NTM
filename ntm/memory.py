"""Implementation of Memory"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
def _convolve(w, s):
    """Implementation of Circular convolution"""
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c


class NTMMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M):
        """Initialize the NTM Memory matrix.
        Dimensions of memory are (batch_size x N x M).
        Every batch has its own memory matrix.
        where N denotes no. of rows and M denotes no. of columns/features in the memory matrix."""
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('mem_bias', torch.Tensor(N, M))

        # Initialize memory bias
        #Random Memory Initialization
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform_(self.mem_bias, -stdev, stdev)
        #Constant memory Initialization
         # nn.init.constant_(self.mem_bias, const_init)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Reading from memory"""
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        """writing to memory"""
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def address(self, k, β, g, s, γ, w_prev):
        """Addressing: 
        This returns a softmax weighting over the rows of the memory matrix.
        where k is The key vector ;
        β is The key strength (focus) ;
        g is Scalar interpolation gate (with previous weighting) ;
        s is Shift weighting ;
        γ is Sharpen weighting scalar ;
        w_prev is The weighting produced in the previous time step."""
        # Content focus
        wc = self._similarity(k, β,w_prev,g)
        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        ŵ = self._shift(wg, s)
        w = self._sharpen(ŵ, γ)

        return w

    def _similarity(self, k, β):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(β * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = torch.zeros(wg.size())
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w

    

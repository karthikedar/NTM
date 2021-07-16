"""NTM Read and Write Heads."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# Spliting a 2D matrix into columns of given lengths
def _split_cols(mat, lengths):
    
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    final_cols = []
    for s, e in zip(l[:-1], l[1:]):
        final_cols += [mat[:, s:e]]
    return final_cols

# Defining NTM read or write head
class NTMHeadBase(nn.Module):
   
    # Initializing NTM read / write head  
    def __init__(self, memory, controller_size):
        # parameter memory: The 'NTMMemory' class to be addressed by the head
        # parameter controller_size: The size of the internal representation
        
        super(NTMHeadBase, self).__init__()

        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size

    def _address_memory(self, k, β, g, s, γ, w_prev):
        # Handle Activations
        # Normalizing the parameters as mentioned in the paper
        k = k.clone()
        β = F.softplus(β)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        γ = 1 + F.softplus(γ)
        # calculating weights corrosponding to memory locations
        weights = self.memory.address(k, β, g, s, γ, w_prev)

        return weights


# Initializing NTM Read head
class NTMReadHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMReadHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ sizes from the paper
        self.read_lengths = [self.M, 1, 1, 3, 1]
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        # The state holds the previous time step address weightings
        return torch.zeros(batch_size, self.N)

    def reset_parameters(self):
        # Initializing the linear layers
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, embeddings, w_prev):
        # NTM read head forward function
        # parameter embeddings: Input from the controller
        # parameter w_prev: weights of previous time step
        
        input_mat = self.fc_read(embeddings)
        k, β, g, s, γ = _split_cols(input_mat, self.read_lengths)

        # Read from the memory and return weights
        weights = self._address_memory(k, β, g, s, γ, w_prev)
        readhead = self.memory.read(weights)

        return readhead, weights


# Initializing NTM Write head
class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMWriteHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ, e, a sizes from the paper
        self.write_lengths = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        # The state holds the previous time step address weightings
        return torch.zeros(batch_size, self.N)

    def reset_parameters(self):
        # Initializing the linear layers
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)
     
    def is_read_head(self):
        # The head is not a read head
        return False

    def forward(self, embeddings, w_prev):
        # NTM write head forward function
        # parameter embeddings: Input from the controller
        # parameter w_prev: weights of previous time step
        input_mat = self.fc_write(embeddings)
        k, β, g, s, γ, e, a = _split_cols(input_mat, self.write_lengths)

        # Setting e in the range from 0 to 1
        e = F.sigmoid(e)

        # Write to the memory
        weights = self._address_memory(k, β, g, s, γ, w_prev)
        self.memory.write(weights, e, a)

        return weights

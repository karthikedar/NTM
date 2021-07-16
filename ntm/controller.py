"""LSTM Controller."""
import torch
from torch import nn
from torch.nn import Parameter
import numpy as np

#torch.set_default_tensor_type(torch.cuda.FloatTensor)

class LSTMController(nn.Module):
    """An NTM controller based on LSTM."""
    def __init__(self, num_inputs, num_outputs, num_layers):
        ''' initialize the LSTM '''
        super(LSTMController, self).__init__()

        """
        Parameters:
        ----------
        input_size: the size of the data input vector
        hidden_size: the size of the hidden state
        num_layers: construct stacked LSTM
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        
        # Controller neural network
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=num_outputs, num_layers=num_layers)

        # Learnable hidden states and cell states
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        
        # Reset parameters
        self.reset_parameters()

    def create_new_state(self, batch_size):
        '''initialize the controller states for the start of a new sequence '''

        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset_parameters(self):
        # Initialize the linear layers
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                '''Initialize the weights by Glorot initialization'''
                torch.nn.init.xavier_uniform_(p, gain=1.0)


    def size(self):
        """retrives the size of the neural network"""
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state):
        '''
        Run a forward path for a sequence 
        Returns the hidden and cell states
        '''
        output, state = self.layer(x.unsqueeze(0), state)
        return output.squeeze(0), state

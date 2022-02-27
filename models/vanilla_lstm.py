"""
Defining the simple LSTM Architecture
"""

import torch.nn as nn

class VLSTM(nn.Module):
    def __init__(self, input_length=2, output_length=2, embed_length=64, hidden_length=64,
                 cell_length=64, dropout_p=0):
        super(VLSTM, self).__init__()

        self.__input_length__ = input_length
        self.__output_length__ = output_length
        self.__embed_length__ = embed_length
        self.__hidden_length__ = hidden_length
        self.__cell_length__ = cell_length
        self.__dropout_p__ = dropout_p

        self.embedding_layer = nn.Sequential(
                nn.Linear(self.__input_length__, self.__embed_length__),
                nn.Tanh(),
                nn.Dropout(p=self.__dropout_p__)
                )

        self.LSTM = nn.LSTMCell(self.__embed_length__, self.__cell_length__)

        self.decoding_layer = nn.Sequential(
                nn.Linear(self.__hidden_length__, self.__output_length__),
                nn.Tanh()
                )
        
    def forward(self, node_pos, edges, 
                edgeRNN_hidden_states, edgeRNN_cell_states, nodeRNN_hidden_state, nodeRNN_cell_state):
        # embed input positions
        embedded = self.embedding_layer(node_pos)
        embedded = embedded.view([1, self.__embed_length__])

        # Run concatenated vector through RNN cell
        h_1, c_1 = self.LSTM(embedded, (nodeRNN_hidden_state, nodeRNN_cell_state))

        # decode hidden state
        output = self.decoding_layer(h_1)

        # return output
        return output, edgeRNN_hidden_states, edgeRNN_cell_states, h_1, c_1

#!/usr/bin/env python

"""
Defining the MESRNN Architecture
"""

__author__ = "Aamir Hasan"
__version__ = "1.0"
__email__ = "hasanaamir215@gmail.com; aamirh2@illinois.edu"

import torch.nn as nn
from torch import sum, stack, cat


class EdgeRNN(nn.Module):
    def __init__(self, input_length=2, embed_length=64,
                 hidden_length=64, cell_length=64, dropout_p=0):
        super(EdgeRNN, self).__init__()

        self.__input_length__ = input_length
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

    def forward(self, inputs, h_0, c_0):
        """
        :param inputs: Input edges of some type
        :type inputs: list of edges, edges of types Tensor(1, 2)
        :param h_0: hidden state for the RNN Cell
        :type h_0: Tensor of size (1, hidden_length)
        :param c_0: cell state for the RNN Cell
        :type c_0: Tensor of size (1, hidden_length)
        :return: (h_1, c_1) the hidden state and cell state after running through input
        :rtype: Tensor of size (1, hidden_length), Tensor of size (1, hidden_length)
        """
        # If there are no edges, just return the input hidden state without doing anything
        if len(inputs) < 1:
            return h_0, c_0

        # average inputs
        averaged_inputs = sum(stack(inputs), dim=0)

        # embed averaged inputs
        embedded = self.embedding_layer(averaged_inputs)
        embedded = embedded.view([1, self.__embed_length__])
        # Run embedded inputs through RNN cell
        h_1, c_1 = self.LSTM(embedded, (h_0, c_0))

        # return output
        return h_1, c_1


class NodeRNN(nn.Module):
    def __init__(self, input_length=2, output_length=2, edges_hidden_length=64, embed_length=64,
                 hidden_length=64, cell_length=64, dropout_p=0, num_edges=6):
        super(NodeRNN, self).__init__()
        
        self.__input_length__ = input_length
        self.__output_length__ = output_length
        self.__edges_hidden_length__ = edges_hidden_length
        self.__embed_length__ = embed_length
        self.__hidden_length__ = hidden_length
        self.__cell_length__ = cell_length
        self.__dropout_p__ = dropout_p
        self.__num_edges__ = num_edges

        self.embedding_layer = nn.Sequential(
                nn.Linear(self.__input_length__, self.__embed_length__),
                nn.Tanh(),
                nn.Dropout(p=self.__dropout_p__)
                )

        self.LSTM = nn.LSTMCell(self.__embed_length__ + self.__edges_hidden_length__ * self.__num_edges__,
                                self.__cell_length__)

        self.decoding_layer = nn.Sequential(
                nn.Linear(self.__hidden_length__, self.__output_length__),
                nn.Tanh()
                )

    def forward(self, node_pos, edges, h_0, c_0):
        """
        :param node_pos: The node trajectory
        :type node_pos: Tensor(1, 2)
        :param edges: List of hidden states from edgeRNNs
        :type edges: List[Tensor(1, edge_hidden_length]
        :param h_0: initial hidden state
        :type h_0: Tensor(1, hidden_length)
        :param c_0: initial cell state
        :type c_0: Tensor(1, hidden_length)
        :return: Next trajectory coordinates, and (h_1, c_1) the hidden state and cell state after running through input
        :rtype: Tensor(1, 2), Tensor of size (1, hidden_length), Tensor of size (1, hidden_length)
        """
        # embed input positions
        embedded = self.embedding_layer(node_pos)
        embedded = embedded.view([1, self.__embed_length__])

        # concatenate embedded vector and hidden states from edges
        concat = cat([embedded, cat(edges, dim=0)], dim=0)
        concat = concat.view([1, self.__embed_length__ + self.__edges_hidden_length__ * self.__num_edges__])
        
        # Run concatenated vector through RNN cell
        h_1, c_1 = self.LSTM(concat, (h_0, c_0))

        # decode hidden state
        output = self.decoding_layer(h_1)

        # return output
        return output, h_1, c_1


class MESRNN(nn.Module):
    def __init__(self, input_length=2, output_length=2, num_edges=6,
                 edgeRNN_embed_length=64, edgeRNN_hidden_length=64, edgeRNN_cell_length=64,
                 nodeRNN_embed_length=64, nodeRNN_hidden_length=64, nodeRNN_cell_length=64,
                 dropout_p=0):
        super(MESRNN, self).__init__()

        self.__input_length__ = input_length
        self.__output_length__ = output_length
        self.__num_edges__ = num_edges

        self.__edgeRNN_embed_length__ = edgeRNN_embed_length
        self.__edgeRNN_hidden_length__ = edgeRNN_hidden_length
        self.__edgeRNN_cell_length__ = edgeRNN_cell_length

        self.__nodeRNN_embed_length__ = nodeRNN_embed_length
        self.__nodeRNN_hidden_length__ = nodeRNN_hidden_length
        self.__nodeRNN_cell_length__ = nodeRNN_cell_length

        self.__dropout_p__ = dropout_p

        self.edgeRNNs = nn.ModuleList()

        for i in range(self.__num_edges__):
            self.edgeRNNs.append(EdgeRNN(self.__input_length__, self.__edgeRNN_embed_length__,
                                         self.__edgeRNN_hidden_length__, self.__edgeRNN_cell_length__,
                                         self.__dropout_p__))

        self.nodeRNN = NodeRNN(self.__input_length__, self.__output_length__, self.__edgeRNN_hidden_length__, 
                                self.__nodeRNN_embed_length__, self.__nodeRNN_hidden_length__, 
                                self.__nodeRNN_cell_length__, self.__dropout_p__, self.__num_edges__)

    def forward(self, node_pos, edges,
                edgeRNN_hidden_states, edgeRNN_cell_states,
                nodeRNN_hidden_state, nodeRNN_cell_state):
        """
        Forward pass for MESRNN

        :param node_pos: Node positions
        :type node_pos: Tensor (1, 2)
        :param edges: List of list of all edges
        :type edges: [[all spatial edges], [all temporal edges], ... ]
        :param edgeRNN_hidden_states: List of all hidden states for EdgeRNNs
        :type edgeRNN_hidden_states: [Tensor(1, edge_hidden_state_length), ... ]
        :param edgeRNN_cell_states: List of all cell states for EdgeRNNs
        :type edgeRNN_cell_states: [Tensor(1, edge_cell_state_length), ... ]
        :param nodeRNN_hidden_state: Hidden state for nodeRNN
        :type nodeRNN_hidden_state: Tensor(1, node_hidden_state_length)
        :param nodeRNN_cell_state: Cell State for nodeRNN
        :type nodeRNN_cell_state: Tensor(1, node_cell_state_length)
        :return: Output Trajectory, edgeRNN_hidden_states, edgeRNN_cell_states, nodeRNN_hidden_states,
                    nodeRNN_cell_states
        :rtype: Tensor(1, 2), [Tensor(1, edge_hidden_state_length), ... ], [Tensor(1, edge_cell_state_length), ... ],
                    Tensor(1, node_hidden_state_length), Tensor(1, node_cell_state_length)
        """
        if self.__num_edges__ != len(edges):
            raise ValueError("Wrong number of edge types configured")

        # Pass all edge types through EdgeRNNs
        output_edgeRNN_hidden_states = []
        output_edgeRNN_cell_states = []

        for i in range(self.__num_edges__):
            h_1, c_1 = self.edgeRNNs[i](edges[i], edgeRNN_hidden_states[i], edgeRNN_cell_states[i])
            output_edgeRNN_hidden_states.append(h_1)
            output_edgeRNN_cell_states.append(c_1)

        # Pass through NodeRNN
        output_delta, output_nodeRNN_hidden_state, output_nodeRNN_cell_state = self.nodeRNN(node_pos,
                                                                                             output_edgeRNN_hidden_states,
                                                                                             nodeRNN_hidden_state,
                                                                                             nodeRNN_cell_state)
        # caluclate output pos from delta
        output_pos = node_pos + output_delta

        return output_pos, \
               output_edgeRNN_hidden_states, output_edgeRNN_cell_states, \
               output_nodeRNN_hidden_state, output_nodeRNN_cell_state

import torch
import torch.nn as nn
from model.SGCRNCell import AGCRNCell

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)



class SGCRN(nn.Module):
    def __init__(self, args, precomputed_embeddings):
        super(SGCRN, self).__init__()
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        if precomputed_embeddings is not None:
            embeddings_tensor = torch.as_tensor(precomputed_embeddings, dtype=torch.float32)
            self.node_embeddings = nn.Parameter(embeddings_tensor, requires_grad=False)
        else:
            self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)
            
        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        # Predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0, 1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN-based predictor
        output = self.end_conv((output))  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output

    #     # Two Encoders for Residual Learning
    #     self.encoder1 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
    #                              args.embed_dim, args.num_layers)
    #     self.encoder2 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
    #                              args.embed_dim, args.num_layers)

    #     # CNN-based Predictors
    #     self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim))
    #     #self.end_conv2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, 12))
    #     self.end_conv2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, min(12, self.hidden_dim)))
    #     self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim))

    # def forward(self, source):
    #     #  Encoder for Major Trend
    #     init_state1 = self.encoder1.init_hidden(source.shape[0])
    #     output1, _ = self.encoder1(source, init_state1, self.node_embeddings)
    #     output1 = output1[:, -1:, :, :]
    #     output1 = self.end_conv1(output1)

    #     #  Compute Residual
    #     source1 = self.end_conv2(output1.permute(0, 3, 2, 1))  # Reshape before Conv2D
    #     source2 = source - source1  # Residual Component
    #     #source2 = (1 - alpha) * (source - source1) + alpha * target  # combining residual and trend
    #     # Step 3: Second Encoder for Residual Learning
    #     init_state2 = self.encoder2.init_hidden(source2.shape[0])
    #     output2, _ = self.encoder2(source2, init_state2, self.node_embeddings)
    #     output2 = output2[:, -1:, :, :]
    #     output2 = self.end_conv3(output2)

    #     return output1 + output2


        
    def update_embeddings(self, new_embeddings):
        """ Update node embeddings dynamically """
        self.node_embeddings.data = torch.tensor(new_embeddings, dtype=torch.float32)
        print("Node embeddings updated!")
        
    def set_embedding_trainable(self, trainable):
        """
        Dynamically update the trainability of node_embeddings.
        Args:
            trainable (bool): Whether to allow gradients to update node_embeddings.
        """
        self.node_embeddings.requires_grad = trainable
        if trainable:
            print("Node embeddings are now trainable.")
        else:
            print("Node embeddings are now frozen.")
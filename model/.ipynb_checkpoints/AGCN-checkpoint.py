import torch
import torch.nn.functional as F
import torch.nn as nn

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv

import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreePhaseFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        """
        Multi-modal traffic feature extractor with two convolutional layers + residual connection.
        """
        super(ThreePhaseFeatureExtractor, self).__init__()

        # First convolutional layer (3×2 Conv2D): Converts 3 channels to 2 channels
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=2, kernel_size=(3,2), padding=(1,0))

        # Second convolutional layer (2×1 Conv2D): Converts 2 channels to 1 channel
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=output_channels, kernel_size=(2,1), padding=(0,0))

    def forward(self, C):
        """
        Args:
            C (torch.Tensor): Input traffic data (B, T, N, 3) -> (batch_size, time_steps, num_nodes, features)

        Returns:
            torch.Tensor: Processed traffic representation (B, T, N, 1)
        """
        B, T, N, F = C.shape  # Extract shape

        # Reshape to match Conv2D input format: (batch, channels, nodes, time_steps)
        C = C.permute(0, 3, 2, 1)  # New shape: (B, 3, N, T)

        # First Convolutional Layer
        C1 = F.relu(self.conv1(C))  # Output shape: (B, 2, N, T)

        # Second Convolutional Layer
        C2 = F.relu(self.conv2(C1))  # Output shape: (B, 1, N, T)

        # Residual Connection: Add traffic flow (first channel) to processed features
        A = C[:, 0:1, :, :]  # Extract only the traffic flow data
        X = A + C2  # Combine extracted traffic flow with convolutional features

        # Reshape back to AGCRN input format: (B, T, N, 1)
        X = X.permute(0, 3, 2, 1)

        return X

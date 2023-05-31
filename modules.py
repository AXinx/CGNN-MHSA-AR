import math
from math import sqrt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back

#
# class FeatureAttentionLayer(nn.Module):
#     """Single Graph Feature/Spatial Attention Layer
#     :param n_features: Number of input features/nodes
#     :param window_size: length of the input sequence
#     :param dropout: percentage of nodes to dropout
#     :param alpha: negative slope used in the leaky rely activation function
#     :param embed_dim: embedding dimension (output dimension of linear transformation)
#     :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
#     :param use_bias: whether to include a bias term in the attention layer
#     """
#
#     def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
#         super(FeatureAttentionLayer, self).__init__()
#         use_cuda =True,
#         self.n_features = n_features
#         self.window_size = window_size
#         self.dropout = dropout
#         self.embed_dim = embed_dim if embed_dim is not None else window_size
#         self.use_gatv2 = use_gatv2
#         self.num_nodes = n_features
#         self.use_bias = use_bias
#
#         self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
#         # Because linear transformation is done after concatenation in GATv2
#         if self.use_gatv2:
#             self.embed_dim *= 2
#             lin_input_dim = 2 * window_size
#             a_input_dim = self.embed_dim
#         else:
#             lin_input_dim = window_size
#             a_input_dim = 2 * self.embed_dim
#
#         self.lin = nn.Linear(lin_input_dim, self.embed_dim)
#         self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#         if self.use_bias:
#             self.bias = nn.Parameter(torch.empty(n_features, n_features))
#
#         self.leakyrelu = nn.LeakyReLU(alpha)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # # x shape (b, n, k): b - batch size, n - window size, k - number of features
#         # # For feature attention we represent a node as the values of a particular feature across all timestamps
#         #
#         #x = x.permute(0, 2, 1)
#         #
#         # # 'Dynamic' GAT attention
#         # # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
#         # # Linear transformation applied after concatenation and attention layer applied after leakyrelu
#         # if self.use_gatv2:
#         #     a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
#         #     a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
#         #     e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)
#         #
#         # # Original GAT attention
#         # else:
#         #     Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
#         #     a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
#         #     e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)
#         #
#         # if self.use_bias:
#         #     e += self.bias
#         #
#         # # Attention weights
#         # attention = torch.softmax(e, dim=2)
#         #
#         # attention = torch.dropout(attention, self.dropout, train=self.training)
#         # print("attention is %s" %(attention))
#         # print(attention.shape)
#         # # Computing new node features using the attention
#         # h = self.sigmoid(torch.matmul(attention, x))
#         # # print("h is %s" %(h))
#         # # print(h.shape)
#         #
#         # # Computing new node features using the attention
#         matrix_all = []
#         y=x.data.cpu().numpy()
#
#         for k in range(y.shape[0]):
#             data = y[k]
#             matrix = np.zeros((data.shape[1], data.shape[1]))
#             for i in range(data.shape[0]):
#                 for j in range(data.shape[1]):
#                     if (i <= j):
#                        matrix[i][j] = np.inner(data[:, i], data[:, j])
#                        #matrix[i][j] = cosine_similarit(np.array(data[:, i]), np.array(data[:, j]))
#                        #  matrix[i][j] = pearsonr(data[:, i], data[:, j])[0]
#                        #  if math.isnan(matrix[i][j]):
#                        #      matrix[i][j] = 0
#                     else:
#                         break
#             matrix = matrix / data.shape[0]
#             matrix_all.append(matrix)
#         attention = torch.from_numpy(np.array(matrix_all))
#         attention=attention.to(dtype=torch.float32)
#         attention=attention.to(self.device)
#
#         h = self.sigmoid(torch.matmul(attention, x.permute(0, 2, 1)))
#         return h.permute(0, 2, 1)
#
#
# class TemporalAttentionLayer(nn.Module):
#     """Single Graph Temporal Attention Layer
#     :param n_features: number of input features/nodes
#     :param window_size: length of the input sequence
#     :param dropout: percentage of nodes to dropout
#     :param alpha: negative slope used in the leaky rely activation function
#     :param embed_dim: embedding dimension (output dimension of linear transformation)
#     :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
#     :param use_bias: whether to include a bias term in the attention layer
#
#     """
#
#     def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
#         super(TemporalAttentionLayer, self).__init__()
#         use_cuda =True,
#         self.n_features = n_features
#         self.window_size = window_size
#         self.dropout = dropout
#         self.use_gatv2 = use_gatv2
#         self.embed_dim = embed_dim if embed_dim is not None else n_features
#         self.num_nodes = window_size
#         self.use_bias = use_bias
#         self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
#         # Because linear transformation is performed after concatenation in GATv2
#         if self.use_gatv2:
#             self.embed_dim *= 2
#             lin_input_dim = 2 * n_features
#             a_input_dim = self.embed_dim
#         else:
#             lin_input_dim = n_features
#             a_input_dim = 2 * self.embed_dim
#
#         self.lin = nn.Linear(lin_input_dim, self.embed_dim)
#         self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#         if self.use_bias:
#             self.bias = nn.Parameter(torch.empty(window_size, window_size))
#
#         self.leakyrelu = nn.LeakyReLU(alpha)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x shape (b, n, k): b - batch size, n - window size, k - number of features
#         # For temporal attention a node is represented as all feature values at a specific timestamp
#
#         # 'Dynamic' GAT attention
#         # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
#         # Linear transformation applied after concatenation and attention layer applied after leakyrelu
#         # if self.use_gatv2:
#         #     a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
#         #     a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
#         #     e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)
#         #
#         # # Original GAT attention
#         # else:
#         #     Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
#         #     a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
#         #     e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)
#         #
#         # if self.use_bias:
#         #     e += self.bias  # (b, n, n, 1)
#         #
#         # # Attention weights
#         # attention = torch.softmax(e, dim=2)
#         # attention = torch.dropout(attention, self.dropout, train=self.training)
#         matrix_all = []
#         y = x.data.cpu().numpy()
#
#         for k in range(y.shape[0]):
#             data = y[k]
#             matrix = np.zeros((data.shape[0], data.shape[0]))
#             for i in range(data.shape[0]):
#                 for j in range(data.shape[1]):
#
#                     matrix[i][j] = np.correlate(data[i, :], data[j, :])
#                     #matrix[i][j] = cosine_similarit(np.array(data[i, :]), np.array(data[j, :]))
#                     # matrix[i][j] = pearsonr(data[:, i], data[:, j])[0]
#                     # if math.isnan(matrix[i][j]):
#                     #     matrix[i][j] = 0
#
#             matrix = matrix / data.shape[0]
#             matrix_all.append(matrix)
#         attention = torch.from_numpy(np.array(matrix_all))
#         attention = attention.to(dtype=torch.float32)
#         attention = attention.to(self.device)
#         h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)

        return h
def TemporalcorrelationLayer(x):
    use_cuda = True  #
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    matrix_all = []
    y = x.data.cpu().numpy()

    for k in range(y.shape[0]):
        data = y[k]
        matrix = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                matrix[i][j] = np.correlate(data[i, :], data[j, :])


        matrix = matrix / data.shape[0]
        matrix_all.append(matrix)
    attention = torch.from_numpy(np.array(matrix_all))
    attention = attention.to(dtype=torch.float32)

    attention = attention.to(device)
    h = torch.sigmoid(torch.matmul(attention, x))  # (b, n, k)

    return h
def FeaturecorrelationLayer(x):
   # print(f'x={x.shape}')
    use_cuda = True  #
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    matrix_all = []
    y = x.data.cpu().numpy()

    for k in range(y.shape[0]):
        data = y[k]
        matrix = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (i <= j):
                    matrix[i][j] = np.inner(data[:, i], data[:, j])
                else:
                    break
        matrix = matrix / data.shape[0]
        matrix_all.append(matrix)
    attention = torch.from_numpy(np.array(matrix_all))
    attention = attention.to(dtype=torch.float32)
    attention=attention.to(device)
   # print(attention.shape)
    h = torch.sigmoid(torch.matmul(attention, x.permute(0, 2, 1)))
    #print(f'h={h.shape}')
    return h.permute(0, 2, 1)


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)


        return self.layers[-1](x)
def Denoising(train):
    use_cuda=True #
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    io_all = []
    for i in range(train.shape[0]):
        data = train[i]
        data = data.data.cpu().numpy()
        io_time = []
        for j in range(data.shape[1]):
            x = data[:, j]
            #x = x.data.cpu().numpy()
            f = np.fft.rfft(x)
            yf_abs = np.abs(f)
            indices = yf_abs > yf_abs.mean()  # filter out those value under 300
            yf_clean = indices * f
            new_f_clean = np.fft.irfft(yf_clean)
            io_time.append(new_f_clean)
        io_time = np.array(io_time)
        io_all.append(io_time)
    io_all = np.array(io_all)
    io_all = torch.from_numpy(np.array(io_all))
    io_all = io_all.to(dtype=torch.float32)
    io_all = io_all.permute(0, 2, 1)
    io_all = io_all.to(device)
    return io_all


class AR(nn.Module):

    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)

        return x


class MHSA(nn.Module):
    def __init__(self, num_heads, dim):
        super().__init__()

        # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.num_heads = num_heads

    def forward(self, x):
        #print(x.shape)
        B, N, C = x.shape
        # 生成转换矩阵并分多头
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # 点积得到attention score
        attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)
        # 乘上attention score并输出
        v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        #print(v.shape)
        return v












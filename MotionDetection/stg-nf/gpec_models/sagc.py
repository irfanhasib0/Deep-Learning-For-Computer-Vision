import torch
import torch.nn as nn
import numpy as np
from gpec_models.graph import Graph

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1,
                 dropout=0,
                 conv_oper = 'sagc',
                 act       = None,
                 out_bn    = True,
                 out_act   = True,
                 residual  = True,
                 headless  = False):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.headless  = headless
        self.out_act   = out_act
        self.act = nn.ReLU(inplace=True) if act is None else act
        if conv_oper == 'sagc' : self.gcn = SAGC(in_channels, out_channels, headless=False)
        if conv_oper == 'gcn'  : self.gcn = ConvTemporalGraphical(in_channels, out_channels)
        #PyGeoConv(in_channels, out_channels, kernel_size=kernel_size[1], dropout=dropout,
        #headless=self.headless, conv_oper=self.conv_oper)

        if out_bn:
            bn_layer = nn.BatchNorm2d(out_channels)
        else:
            bn_layer = nn.Identity()  # Identity layer shall no BN be used

        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 self.act,
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (kernel_size[0], 1),
                                           (stride, 1),
                                           padding),
                                 bn_layer,
                                 nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()

        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                          out_channels,
                                          kernel_size=1,
                                          stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x, adj):
        if type(self.residual)==int: res=0
        else : res = self.residual(x)
        x, adj = self.gcn(x, adj)
        x = self.tcn(x) + res
        if self.out_act:
            x = self.act(x)

        return x, adj

    
class SAGC(nn.Module):
    """
    Spatial Attention Graph Convolution
    Applied to K_n adjacency subsets, each with K_a matrices
    Base class provides the data-based C matrix and returns
    K_n * K_a results.
    """
    def __init__(self, in_channels, out_channels,
                 stat_adj=None,
                 num_subset=3,
                 coff_embedding=4,
                 ntu=False,
                 headless=False):
        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.adj_types = 3
        self.inter_c = inter_channels
        if stat_adj is None:  # Default value for A:
            layout = 'ntu-rgb+d' if ntu else 'openpose'
            self.graph = Graph(strategy='spatial', layout=layout, headless=headless)
            stat_adj = self.graph.A

        self.g_adj = torch.from_numpy(stat_adj.astype(np.float32))
        nn.init.constant_(self.g_adj, 1e-6)
        self.g_adj = nn.Parameter(self.g_adj, requires_grad=True)

        self.stat_adj = torch.from_numpy(stat_adj.astype(np.float32))
        self.stat_adj.requires_grad = False
        self.num_subset = num_subset

        # Convolutions for calculating C adj matrix
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))

        self.gconv = nn.ModuleList()
        for i in range(self.adj_types):
            self.gconv.append(GraphConvBR(out_channels))

        # Residual Layer
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                      nn.BatchNorm2d(out_channels))
        else:
            self.down = nn.Identity()

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)  # Cols sum to 1.0, perhaps rows should?
        self.relu = nn.CELU(0.01)  # nn.ReLU()

        self.kernel_size = self.stat_adj.size(0)  # Split to two different operators?

        self.kernel_size *= 3  # Separate channels for each adj matrix
        self.expanding_conv = nn.Conv2d(in_channels,
                                        out_channels * self.kernel_size,
                                        kernel_size=(1, 1),
                                        padding=(0, 0),
                                        stride=(1, 1),
                                        dilation=(1, 1),
                                        bias=False)

        self.reduction_conv = nn.Conv2d(3 * out_channels, out_channels, 1, bias=False)

    def forward(self, x, adj):
        C = self.calc_C(x)
        x_residual = self.down(x)
        x = self.expanding_conv(x)

        A = adj
        N, KC, T, V = x.size()
        x = x.view(N, self.kernel_size, KC//self.kernel_size, T, V)

        # Each gconv gets out_channels as input, for sep_sum the channels
        # are from separate convolution kernels, otherwise they are the same inputs
        x_A = self.gconv[0](x[:, 0:3], A)
        x_B = self.gconv[1](x[:, 3:6], self.g_adj)
        x_C = self.gconv[2](x[:, 6:9], C)

        # Summarize over A0,A1,A2, and conv-pool over x_A, x_B, x_C
        x = torch.cat((x_A, x_B, x_C), dim=1)
        x = self.reduction_conv(x)
        y = x + x_residual
        return y, adj

    def calc_C(self, x):
        N, chan, T, V = x.size()
        # Calc C, the data-dependent adjacency matrix
        C = []
        for i in range(self.num_subset):
            C1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            C2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            C_curr = self.soft(torch.matmul(C1, C2))  # / C1.size(-1))  # N V V
            C.append(C_curr)
            del C1, C2

        C = torch.stack(C).permute(1, 0, 2, 3).contiguous()
        return C


class GraphConvBR(torch.nn.Module):
    def __init__(self, out_channels):
        """
        Applies a single adjacency graph convolution with BN and Relu
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, adj):
        if len(adj.size()) == 3:  # Global adj matrix
            x = torch.einsum('nkctv,kvw->nctw', (x, adj))
        elif len(adj.size()) == 4:  # Per sample matrix
            x = torch.einsum('nkctv,nkvw->nctw', (x, adj))
        x = self.act(self.bn(x))
        return x

class ConvTemporalGraphical(nn.Module):

    """
    The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 t_kernel_size=1,
                 t_stride=1,
                 bias=True):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            stride=(t_stride, 1),
            dilation=(1, 1),
            bias=bias)

    def forward(self, x, adj):
        self.kernel_size = 1#adj.size(0)
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, adj))

        return x.contiguous(), adj

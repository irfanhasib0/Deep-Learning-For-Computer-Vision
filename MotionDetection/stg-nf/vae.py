import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from copy import copy
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple

from nf_models.stgcn import st_gcn
from nf_models.graph import Graph


Tensor = TypeVar('torch.tensor')


class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


def vae_loss(recons,inps,mu,log_var) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        kld_weight = 0.00025 #kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, inps)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
    
def ae_loss(recons,inps) -> dict:
        recons_loss =F.mse_loss(recons, inps)
        return {'loss': recons_loss, 'Reconstruction_Loss':recons_loss.detach()}

"""
The based unit of graph convolutional networks., based on awesome previous work by https://github.com/yysijie/st-gcn
"""

def stg_nf_loss(z,nll):
    return {'loss': torch.mean(nll)}
    

class StGcn(nn.Module):
    
    def __init__(self,
                 arch,
                 in_channels,
                 out_channels,
                 kernel_size,
                 residual      = True,
                 num_heads     = 1,
                 num_embd_lyrs = 1,
                 num_deep_lyrs = 1,
                 num_of_kps    = 18):
        super(StGcn,self).__init__()
        
        self.arch = arch
        
        if residual:
            self.residual = self.build_residual_layer(in_channels,out_channels,ksize=1)
        else:
            self.residual = lambda x : 0
            
        t_dilation = [1,2,3]
        self.layer_embd = []
        self.layer_deep = []
        for i in range(num_embd_lyrs):
            self.layer_embd.append(self.build_embedding_layer(in_channels,out_channels,kernel_size,t_dilation[i]))
        self.layer_embd = nn.ModuleList(self.layer_embd)
            
        self.se_ratio = 1 
        self.conv_qry = []
        self.conv_key = []
        self.conv_exp = []
        self.conv_val = []
        for _ in range(num_heads):
            self.conv_qry.append(nn.Conv2d(out_channels,out_channels//self.se_ratio,(1,1)))
            self.conv_key.append(nn.Conv2d(out_channels,out_channels//self.se_ratio,1))
            if self.se_ratio > 1:
                self.conv_exp.append(nn.Conv2d(out_channels//self.se_ratio,out_channels,1))
            self.conv_val.append(nn.Conv2d(out_channels,out_channels,1))
        
        self.conv_qry = nn.ModuleList(self.conv_qry)
        self.conv_key = nn.ModuleList(self.conv_key)
        self.conv_exp = nn.ModuleList(self.conv_exp)
        self.conv_val = nn.ModuleList(self.conv_val)
        self.adjs     = nn.Parameter(torch.ones((num_heads,num_of_kps,num_of_kps),dtype=torch.float32))
        
        for _ in range(num_deep_lyrs):
            self.layer_deep.append(self.build_deeper_layer(out_channels,out_channels,3,t_dilation[i]))
        self.layer_deep = nn.ModuleList(self.layer_deep)
        
        '''
        self.conv_t = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        '''
    def build_residual_layer(self,in_channels,out_channels,ksize=1,stride=1):
        return  nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=ksize,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
    
    def build_embedding_layer(self,in_channels,out_channels,ksize,dilation=1,stride=1):
        if self.arch == 'enc' or self.arch == 'dec':
            return nn.Sequential(
            #nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size = (ksize, 1),
                padding     = (dilation*(ksize-1)//2, 0),
                stride      = (stride, 1),
                dilation    = (dilation, 1),
                bias        = True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
        if self.arch == '_dec':
            return nn.Sequential(
            #nn.BatchNorm2d(in_channels),
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size = (ksize, 1),
                padding     = (dilation*(ksize-1)//2, 0),
                stride      = (stride, 1),
                dilation    = (dilation, 1),
                bias        = True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
    
    def build_deeper_layer(self,in_channels,out_channels,ksize,dilation=1,stride=1):
        return nn.Sequential(
        #nn.BatchNorm2d(out_channels),
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = (ksize, ksize),
            padding     = (dilation*(ksize-1)//2, dilation*(ksize-1)//2),
            stride      = (stride, 1),
            dilation    = (dilation, 1),
            bias        = True),
        nn.BatchNorm2d(out_channels),
        #nn.ReLU()
        )
     
    def get_edge_attention(self, x,i,A=None):
        if A == None:
            N, Ch, T, V = x.size()
            A1 = self.conv_qry[i](x).permute(0, 1, 3, 2)
            A2 = self.conv_key[i](x)

            A  = torch.einsum('ncvt,nctw->ncvw',(A1, A2))
            #A  = nn.ReLU()(A)
            if self.se_ratio> 1:
                A  = self.conv_exp[i](A)#.view(N, self.inter_c * T, V)
            A  = nn.Softmax(dim=1)(A)  # / C1.size(-1))  # N V V
        
        #x  = self.conv_val[i](x)
        if len(A.shape)==4:
            x  = torch.einsum('nctv,ncvw->nctw', (x, A))  #'nkctv,kvw->nctw'
        else:
            A  = nn.Softmax(dim=0)(A)
            x  = torch.einsum('nctv,vw->nctw', (x, A))  #'nkctv,kvw->nctw'
        return x
    
    def forward(self, x, A):
        res = self.residual(x)
        for i in range(len(self.layer_embd)):
            x   = self.layer_embd[i](x)
        #n, c, t, v = x.size()
        #x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        #x = x.view(n, c, t, v)
        for i in range(len(self.conv_qry)):
            x = self.get_edge_attention(x,i,A=self.adjs[i])
            
        for i in range(len(self.layer_deep)):
            x = self.layer_deep[i](x) + res
        return x.contiguous() + res
'''

class StGcn(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x   = self.gcn(x, A) +res
        #x = self.tcn(x) + res

        return self.relu(x)

'''    
class ConvModel(nn.Module):
    def __init__(self,in_channels,hidden_dims,mode='enc'):
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, out_channels=h_dim,
                      kernel_size= 3, stride= 1, padding  = 1),
            nn.BatchNorm2d(h_dim),
            nn.LeakyReLU())
    
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[i],
                               hidden_dims[i + 1],
                               kernel_size=3,
                               stride = 1,
                               padding=1),
            nn.BatchNorm2d(hidden_dims[i + 1]),
            nn.LeakyReLU())
        
        self.fwd = self.enc if mode == 'enc' else self.dec
        
    def forward(self,x):
        return self.fwd(x)
'''
class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
'''        
class VAE(BaseVAE):
    def __init__(self,
                 args,
                 in_channels: int,
                 hidden_dims: List = None) -> None:
        super(VAE, self).__init__()
        self.device = args['device']
        self.V      = args['no_of_kps']
        self.T      = args['seg_len']
        self.ksize  = args['ksize']
        self.latent_dim  = args['latent_dim']
        self.num_heads  = args['num_heads']
        self.num_embd_lyrs = args['num_embd_lyrs']
        self.num_deep_lyrs = args['num_deep_lyrs']
        self.set_seed(args['seed'])
        
        self.adjs = torch.concat([self.get_adj(i) for i in range(2,len(hidden_dims)+2)],dim=0)
        self.adjs = Parameter(self.adjs)
        
        self.hidden_dims = copy(hidden_dims)
        self.encoder     = self.build_encoder(in_channels,hidden_dims)
        
        self.t       = self.T#(self.T - len(self.hidden_dims)*(self.ksize - 1))
        out_size     = hidden_dims[-1] * self.V * self.t
        self.fc_mu   = nn.Linear(out_size, self.latent_dim)
        self.fc_var  = nn.Linear(out_size, self.latent_dim)
        
        self.decoder_input = nn.Linear(self.latent_dim, out_size)
        self.decoder = self.build_decoder(in_channels,hidden_dims)
        
        self.final_layer = nn.Sequential(
                           #nn.ConvTranspose2d(hidden_dims[-1],
                           #                    hidden_dims[-1],
                           #                    kernel_size=3,
                           #                    stride=1,
                           #                    padding=1),
                           #nn.BatchNorm2d(hidden_dims[-1]),
                           #nn.LeakyReLU(),
                           #nn.Conv2d(hidden_dims[-1], out_channels= in_channels,
                           #          kernel_size= 3, padding= 1),
                           nn.Tanh())
        
    def get_adj(self,hop):
        A = Graph(strategy='uniform', max_hop=hop).A
        A = torch.tensor(A,dtype=torch.float32,requires_grad=True)
        A = A.view(1,1,self.V,self.V).to(self.device)
        #self.A = self.A.repeat([3,1,1])
        A = torch.ones((1,1,self.V,self.V),dtype=torch.float32).to(self.device)
        #A = Parameter(A)
        return A
    
    def set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        #torch.use_detecministic_algorithms(seed)
    
    def build_encoder(self,in_channels,hidden_dims):
        modules = []
        in_ch = in_channels
        for i,h_dim in enumerate(hidden_dims):
            modules.append(
                StGcn('enc',
                      in_ch,
                      h_dim,
                      self.ksize,
                      residual=not (i==0),
                      num_heads     = self.num_heads,
                      num_embd_lyrs = self.num_embd_lyrs,
                      num_deep_lyrs = self.num_deep_lyrs)
            )
            in_ch = h_dim
            
        return nn.ModuleList(modules)
    
    def build_decoder(self,out_channel,hidden_dims):
        modules = []
        hidden_dims.reverse()
        hidden_dims += [out_channel]
        #output_padding=1),
        residual = False
        for i in range(len(hidden_dims) - 1):
            modules.append(
                StGcn('dec',
                      hidden_dims[i],
                      hidden_dims[i+1],
                      self.ksize,
                      residual= not (i==0),
                      num_heads     = self.num_heads,
                      num_embd_lyrs = self.num_embd_lyrs,
                      num_deep_lyrs = self.num_deep_lyrs)
            )

        return nn.ModuleList(modules)
     
    def encode(self, inp: Tensor) -> List[Tensor]:
        result = inp
        for i,enc in enumerate(self.encoder):
            result = enc(result,self.adjs[i])
        #result = self.encoder(inp)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distributions
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.t , self.V)
        for i,dec in enumerate(self.decoder):
            j = len(self.adjs)-(i+1)
            result = dec(result,self.adjs[j])
        #result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
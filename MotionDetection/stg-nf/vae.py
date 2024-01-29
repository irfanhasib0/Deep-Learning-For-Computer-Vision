import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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
    
class VAE(BaseVAE):
    def __init__(self,
                 args,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None) -> None:
        super(VAE, self).__init__()
        self.set_seed(args['seed'])
        self.V = args['no_of_kps']
        self.T = args['seg_len']
        self.latent_dim = latent_dim
        self.A = Graph(strategy='uniform', max_hop=8).A
        self.A = torch.tensor(self.A,dtype=torch.float32).to(args['device'])
        self.A = self.A.repeat([3,1,1])
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        
        self.hidden_dims = copy(hidden_dims)
        self.encoder     = self.build_encoder(in_channels,hidden_dims)
        
        self.fc_mu   = nn.Linear(hidden_dims[-1]*self.V*self.T, latent_dim)
        self.fc_var  = nn.Linear(hidden_dims[-1]*self.V*self.T, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]* self.V*self.T)
        self.decoder = self.build_decoder(latent_dim,hidden_dims)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        
    '''
    def build_encoder(self,in_channels,hidden_dims):
        modules = []
        in_ch = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels=h_dim,
                              kernel_size= 3, stride= 1, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_ch = h_dim
        
        return nn.Sequential(*modules)
    
    def build_decoder(self,latent_dim,hidden_dims):
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]* 24*18)
        hidden_dims.reverse()
        #output_padding=1),
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 1,
                                       padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        return nn.Sequential(*modules)
    '''
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
                st_gcn(in_ch,h_dim,(9,3),residual=not (i==0))
            )
            in_ch = h_dim
            
        return nn.ModuleList(modules)
    
    def build_decoder(self,latent_dim,hidden_dims):
        modules = []
        hidden_dims.reverse()
        #output_padding=1),
        residual = False
        for i in range(len(hidden_dims) - 1):
            modules.append(
                st_gcn(hidden_dims[i],hidden_dims[i+1],(9,3),residual= not (i==0))
            )

        return nn.ModuleList(modules)
    
        
    def encode(self, inp: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = inp
        for enc in self.encoder:
            result,_ = enc(result,self.A)
        #result = self.encoder(inp)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distributions
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.T, self.V)
        for dec in self.decoder:
            result,_ = dec(result,self.A)
        #result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        input
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
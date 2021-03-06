
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from likelihoods.likelihood import Likelihood
from torch.autograd import Variable


# []

class Gaussian(Likelihood):
    """
    Class for Gaussian Likelihood
    """
    def __init__(self, Dc, L, K):
        super(Gaussian, self).__init__()

        # Gaussian parameters initialization
        self.K = K
        self.Dc = Dc
        self.L = L
        self.W_c = torch.nn.Parameter(torch.rand((self.K, self.Dc, self.L)),requires_grad=True)#.double()
        var_ini = torch.Tensor(np.random.uniform(0.01,5,size=(self.K,self.Dc,1)))
        self.var_c = torch.nn.Parameter(var_ini,requires_grad=True)
        #self.var_c = torch.nn.Parameter((0.05 - 0.5) * torch.rand(self.K,self.Dc,1) + 0.5,requires_grad=True)
        #self.covMat_c = torch.nn.Parameter(torch.uniform(0.05,0.1)*torch.eye(self.Dc),requires_grad=True)#.double()


    def log_pdf(self,k,x, z_pred):

        x = torch.from_numpy(x).double()
        # z = q_pdf.sample().double()
        z = z_pred.double()

        #print(torch.matmul(self.W_c[k,:,:].double(),z.t()).shape)
        #print((self.var_c[k,:]*torch.eye(self.Dc)).shape)

        p = MultivariateNormal(torch.matmul(self.W_c[k,:,:].double(),z.t()).t(),self.var_c[k,:]*torch.eye(self.Dc).double())
        self.logpdf = p.log_prob(x)


        return torch.mean(self.logpdf)


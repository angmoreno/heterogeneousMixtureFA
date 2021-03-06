import torch
from likelihoods.likelihood import Likelihood
import torch.distributions as distributions
import numpy as np
# []

class Bernoulli(Likelihood):
    """
    Class for Bernoulli Likelihood
    """
    def __init__(self,Dd,L, K):
        super(Bernoulli, self).__init__()

        # Bernoulli parameters initialization
        self.K = K
        self.Dd = Dd
        self.L = L
        self.W_d = torch.nn.Parameter(torch.rand((self.K,self.Dd, self.L)),requires_grad=True)#.double()
        alpha = np.random.uniform(0.25,0.75,size=(self.Dd,self.K))
        self.mu_d = torch.nn.Parameter(torch.Tensor(alpha),requires_grad=True)#.double()


    def log_pdf(self,k, x, z_pred):

        x = torch.from_numpy(x).double()
        z = z_pred.double()

        self.eta_softmax = torch.nn.functional.softmax(torch.matmul(self.W_d[k,:,:].double(),z.t()).t() + self.mu_d[:,k].double(), dim=0)

        p = distributions.bernoulli.Bernoulli(self.eta_softmax)

        self.logpdf = p.log_prob(x)


        return torch.mean(self.logpdf)




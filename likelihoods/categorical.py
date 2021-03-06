import torch
from likelihoods.likelihood import Likelihood
import torch.distributions as distributions
import numpy as np

# []

class Categorical(Likelihood):
    """
    Class for Categorical Likelihood
    """
    def __init__(self,Dd,L, K):
        super(Categorical, self).__init__()

        # Categorical parameters initialization

        self.K = K
        self.Dd = Dd
        self.L = L
        self.W_d = torch.nn.Parameter(torch.rand((self.K, self.Dd, self.L)),requires_grad=True)#.double()
        #self.mu_d = torch.nn.Parameter(torch.rand(self.Dd),requires_grad=True)#.double()
        #alpha = np.random.dirichlet(self.Dd*[0.5],self.K).transpose()
        alpha = np.random.uniform(0.6,0.75,size=(self.Dd,self.K))

        self.mu_d = torch.nn.Parameter(torch.Tensor(alpha),requires_grad=True)


    def log_pdf(self,k, x, z_pred):

        x = torch.from_numpy(x).double()
        z = z_pred.double()

        self.eta_softmax = torch.nn.functional.softmax(torch.matmul(self.W_d[k,:,:].double(),z.t()).t() + self.mu_d[:,k].double(), dim=0)  #de mu_d quitar 0 poner K

        p = distributions.categorical.Categorical(self.eta_softmax.t())
        try:
            self.logpdf = p.log_prob(x)
        except:
            return 0

        return torch.mean(self.logpdf)




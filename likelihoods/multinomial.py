import torch
from likelihoods.likelihood import Likelihood
import torch.distributions as distributions

# []

class Multinomial(Likelihood):
    """
    Class for Multinomial Likelihood
    """
    def __init__(self,Dd,M,L,K):
        super(Multinomial, self).__init__()

        # Multinomial parameters initialization

        self.Dd = Dd
        self.M = M
        self.L = L
        self.W_d = torch.nn.Parameter(torch.rand((self.Dd, self.L, self.M + 1)),requires_grad=True)#.double()
        self.mu_d = torch.nn.Parameter(torch.rand(self.Dd),requires_grad=True)#.double()


    def log_pdf(self,x, z_pred):

        x = torch.from_numpy(x).double()
        z = z_pred.double()

        mu = torch.zeros(self.M+1,self.Dd)
        p_m = torch.zeros(self.M+1,self.Dd)
        multinom_logpdf = 0

        S = 10e4


        for m in range(self.M+1):

            mu[m, :] = torch.matmul(self.W_d[:,:,m].double(), z) + self.mu_d.double()
            #mu_softmax[m, :] = torch.nn.functional.softmax(mu[m,:], dim=0)
            #p_m = distributions.multinomial.Multinomial(probs=mu_softmax[m,:])
            #p_m = distributions.multinomial.Multinomial(probs=torch.nn.functional.softmax(mu[m,:], dim=0))
            p_m = distributions.multinomial.Multinomial(probs=torch.nn.functional.softmax( torch.matmul(self.W_d[:,:,m].double(), z) + self.mu_d.double(), dim=0))
            multinom_logpdf += p_m.log_prob(x[:,m])

        self.logpdf = multinom_logpdf  # Montecarlo estimates


        return self.logpdf




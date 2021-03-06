import math
import torch
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import multinomial
from torch.distributions.normal import Normal
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multinomial import Multinomial
from torch.distributions.categorical import Categorical

# []


class Mixture_Model(torch.nn.Module):
    def __init__(self,N, L,likelihoods):
        super(Mixture_Model, self).__init__()

        self.likelihoods = likelihoods
        self.gaussian = self.likelihoods[0]
        self.bernoulli = self.likelihoods[1]
        self.categorical = self.likelihoods[2]


        self.K = self.likelihoods[0].K
        self.D = self.likelihoods[0].Dc
        self.L = L
        self.N = N

        # Variational distribution q(z|x,theta) -- mean + variance  (INITALIZATION)
        self.q_z_mean = torch.nn.Parameter(torch.rand((self.N, self.L)),requires_grad=True)#.double()
        self.q_log_var = torch.nn.Parameter(torch.rand((self.N, self.L)),requires_grad=True)

        # Variational distribution q(s|x,theta) -- probs (INITALIZATION)
        aux_mu_s = torch.Tensor(np.random.dirichlet(np.ones(self.K)*0.53*self.D,self.N))
        self.q_s_param = torch.nn.Parameter(aux_mu_s, requires_grad=True)
        #self.q_s_param = torch.nn.Parameter(torch.ones((self.N, self.K))/self.K, requires_grad=True)

        # True posterior distribution p(z|theta) -- N(0|1)
        self.posterior_mean = torch.nn.Parameter(torch.zeros((self.N,self.L)), requires_grad=False)#.double()
        self.posterior_var = torch.nn.Parameter(torch.ones((self.N,self.L)), requires_grad=False)#.double()

        # True posterior distribution p(s) -- Categorical -- Cat(posterior_mu)
        self.posterior_mu = torch.nn.Parameter(torch.ones((self.N, self.K))/self.K, requires_grad=False)


        '''
        self.posterior_probs = torch.nn.Parameter(torch.Tensor((1/self.K)*np.ones((1,self.N))),requires_grad = False)
        # Mixture coefficients (Weight of each class)
        self.pi = (1 / self.K) * np.ones((1, self.K))

        # Expected z
        self.r_ik = np.ones((self.N, self.K)) / self.K
        '''


    def reparametrize(self, q_mean, q_var):

        logvar = torch.log(torch.diagonal(q_var))
        sigma = torch.exp(0.5*logvar)
        eps = torch.randn_like(sigma) # N(0,I) with same size than sigma
        z_pred = q_mean + sigma * eps
        return z_pred


    def forward(self, index, X_c=None, X_d=None, X_b=None):

        B = len(X_c) #shape of batch
        M_samples = 1000

        KL_z = 0
        KL_s = 0
        LL = 0
        elbo = 0

        term_1 = 0
        term_2 = 0
        term_3 = 0

        rik = np.zeros((self.N,self.K))

        start = B * index
        end = B * (index+1)

        for i in range(start,end):
        #for i in range(len(X_c)):
            # -------Variational function q(z)-------
            self.q_z_CoVar = torch.exp(self.q_log_var[i,:]) * torch.eye(self.L)
            q_z = MultivariateNormal(self.q_z_mean[i,:], self.q_z_CoVar)
            p_z = MultivariateNormal(self.posterior_mean[i,:], self.posterior_var[i,:]*torch.eye(self.L))
            KL_i_z = torch.distributions.kl.kl_divergence(q_z, p_z)
            #print(f'KL z = {KL_i_z}')
            # -------Variational function q(s)-------
            q_s = Categorical(self.q_s_param[i,:])
            p_s = Categorical(self.posterior_mu[i,:])
            KL_i_s = torch.distributions.kl.kl_divergence(q_s, p_s)
            s_pred = torch.nn.functional.gumbel_softmax(self.q_s_param)  # Esto me daria el perfil estimado para ese dia
            #print(f'KL s = {KL_i_s}')
            # ----------------------------------------

            LL_i = 0
            term_1_i = 0
            term_2_i = 0
            term_3_i = 0
            z_pred = q_z.rsample([M_samples])  # M x L


            for k in range(self.K):
                term_1_k = self.gaussian.log_pdf(k, X_c[i-start], z_pred)
                term_2_k = self.bernoulli.log_pdf(k, X_b[i-start], z_pred)
                term_3_k = self.categorical.log_pdf(k, X_d[i-start], z_pred)
                #term_3_k = 0
                LL_i += self.q_s_param[i,k] * torch.sum(term_1_k + term_2_k + term_3_k)
                term_1_i += term_1_k
                term_2_i += term_2_k
                term_3_i += term_3_k

            rik[i,:] = torch.nn.functional.softmax(self.q_s_param[i,:],dim=0).detach().numpy()
            elbo_i = LL_i - KL_i_z - KL_i_s

            elbo += elbo_i

           # print(KL_s)
            term_1 += term_1_i
            term_2 += term_2_i
            term_3 += term_3_i
            LL += LL_i

        KL_z = KL_i_z
        KL_s = KL_i_s

        return -elbo,LL,KL_z, KL_s, rik, term_1,term_2,term_3












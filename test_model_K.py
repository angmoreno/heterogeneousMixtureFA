
import toy_distributions as dist
import util
import torch
import numpy as np
import matplotlib
#matplotlib.use("Pdf")
import matplotlib.pyplot as plt
from model import Mixture_Model
import model
from likelihoods.gaussian import Gaussian
from likelihoods.multinomial import Multinomial
from likelihoods.bernoulli import Bernoulli
from likelihoods.categorical import Categorical
from torch import autograd


def test_model(N,Dc,Dd,Db,L,K,X_c=None,X_d=None,X_b=None):

    batch_size = int(N/4)
    epsilon = 1e0

    #  ----------- Model ------------
    gaussian = Gaussian(Dc, L, K)
    categorical = Categorical(Dd, L, K)
    bernoulli = Bernoulli(Db, L, K)

    likelihoods = [gaussian,bernoulli,categorical]

    model = Mixture_Model(N, L, likelihoods)
    optim = torch.optim.Adagrad(model.parameters(), lr=0.01)
    autograd.set_detect_anomaly(True)
    # optim = torch.optim.SGD(model.parameters(),lr=0.001, momentum= 0.9)

    data_set = torch.utils.data.TensorDataset(torch.Tensor(X_c), torch.Tensor(X_d),torch.Tensor(X_b))
    #data_set = torch.utils.data.TensorDataset(torch.Tensor(X_c),torch.Tensor(X_b))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)  # shuffle a true?
    #data_loader= torch.utils.data.DataLoader(X_c, batch_size = batch_size, shuffle=False)  #shuffle a true?

    num_epochs = 100
    ll_list = []
    loss_list = []
    KL_z_list = []
    KL_s_list = []
    rik_epochs = []
    term_1_list = []
    term_2_list = []
    term_3_list = []

    past_loss = 0

    for epoch in range(num_epochs):

        loss_epoch = 0
        ll_epoch = 0
        KL_z_epoch = 0
        KL_s_epoch = 0
        term_1_epoch = 0
        term_2_epoch = 0
        term_3_epoch = 0

        # for x_batch_real, x_batch_discrete in data_loader:
        for index, x_batch in enumerate(data_loader):
            x_batch_real = x_batch[0]
            x_batch_disc = x_batch[1]
            x_batch_bin = x_batch[2]


            # ----- Variational E ----- fix θ
            optim.zero_grad()
            util.fix_model_params(likelihoods, set=False)
            util.fix_variational_params(model, set=True)
            loss, LL, KL_z, KL_s, rik, term_1, term_2,term_3  = model(index, X_c=x_batch_real.numpy(), X_d=x_batch_disc.numpy(), X_b=x_batch_bin.numpy())
            loss.backward()
            optim.step()

            # ----- Variational M ----- fix φ

            optim.zero_grad()
            util.fix_model_params(likelihoods, set=True)
            util.fix_variational_params(model, set=False)
            loss, LL, KL_z, KL_s, rik, term_1, term_2,term_3  = model(index, X_c=x_batch_real.numpy(), X_d=x_batch_disc.numpy(), X_b=x_batch_bin.numpy())
            loss.backward()
            optim.step()
            ll_epoch += LL
            KL_s_epoch += KL_s
            KL_z_epoch += KL_z
            loss_epoch += loss

            term_1_epoch += term_1
            term_2_epoch += term_2
            term_3_epoch += term_3



        #print(f"Epoch = {epoch}, Loglik ={ll_epoch}, -ELBO ={loss_epoch}")
        rik_epochs.append(rik)
        KL_z_list.append(KL_z_epoch)
        KL_s_list.append(KL_s_epoch)
        loss_list.append(loss_epoch)
        term_1_list.append(term_1_epoch)
        term_2_list.append(term_2_epoch)
        term_3_list.append(term_3_epoch)
        ll_list.append(ll_epoch)


    z_mean = model.q_z_mean
    W_c = model.gaussian.W_c
    var_c =model.gaussian.var_c
    W_b = model.bernoulli.W_d
    W_d = model.categorical.W_d
    #W_d = None
    mu_d = model.categorical.mu_d
    #mu_d = None
    mu_b = model.bernoulli.mu_d
    param = torch.nn.functional.softmax(model.q_s_param, dim=1).detach().numpy()
    #print(param)

    profiles = np.argmax(param, axis=1) + 1

    '''
    plt.figure()
    plt.plot(np.arange(num_epochs), KL_z_list)
    plt.title(f'Convergence of KL_z for K={K}')
    plt.xlabel('Epochs')
    plt.ylabel('Kullback-Leibler divergence')
    plt.savefig('KL_z_'+str(K)+'.png')

    plt.figure()
    plt.plot(np.arange(num_epochs), KL_s_list)
    plt.title(f'Convergence of KL_s for K={K}')
    plt.xlabel('Epochs')
    plt.ylabel('Kullback-Leibler divergence')
    plt.savefig('KL_s_'+str(K)+'.png')

    '''

    plt.figure()
    plt.plot(np.arange(num_epochs), term_1_list)
    plt.title(f'Convergence of ELBO terms for K={K}')
    plt.legend([ 'Gaussian Term '])
    plt.xlabel('Epochs')
    plt.ylabel('Likelihood')
    plt.savefig('GaussianTerm_'+str(K)+'.png')



    plt.figure()
    plt.plot(np.arange(num_epochs), term_2_list)
    plt.title(f'Convergence of ELBO terms for K={K}')
    plt.legend(['Bernoulli term'])
    plt.xlabel('Epochs')
    plt.ylabel('Likelihood')
    plt.savefig('BernoulliTerm_'+str(K)+'.png')

    plt.figure()
    plt.plot(np.arange(num_epochs), term_3_list)
    plt.title(f'Convergence of ELBO terms for K={K}')
    plt.legend(['Categorical term'])
    plt.xlabel('Epochs')
    plt.ylabel('Likelihood')
    plt.savefig('CategoricalTerm_'+str(K)+'.png')


    plt.figure()
    plt.plot(np.arange(num_epochs), ll_list)
    plt.plot(np.arange(num_epochs), loss_list)
    plt.title(f'Performance in epochs for K={K}')
    plt.legend(['Likelihood evolution', 'Loss evolution'])
    plt.xlabel('Epochs')
    plt.ylabel('Likelihood')
    plt.savefig('Convergence_'+str(K)+'.png')

    #plt.show()


    return ll_list[-1],z_mean,W_c,W_b,mu_b,mu_d,W_d,var_c,profiles
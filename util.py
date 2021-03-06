import numpy as np
import scipy
import bayespy.nodes as nodes
import read_dataset
import sklearn.metrics as metrics
import random
import torch
import torch.distributions as distributions

#from scipy.special import softmax

# []


def onehot(Y_dis,M):


    N = Y_dis.shape[0]
    Dd = Y_dis.shape[1]

    Y_onehot = np.zeros((N,Dd,int(M+1)))

    for i in range(N):
        a = Y_dis[i]
        b = np.zeros((a.size, M+1))

        for fila in range(a.size):
            b[fila][int(a[fila])]=1

        Y_onehot[i,:,:] = b

    return Y_onehot


def fix_model_params(likelihoods, set=False):
    gaussian = likelihoods[0]
    gaussian.W_c.requires_grad = set
    gaussian.var_c.requires_grad = set

    if len(likelihoods)>1:
        discrete = likelihoods[1]
        discrete.W_d.requires_grad = set
        discrete.mu_d.requires_grad = set
    elif len(likelihoods)>2:
        discrete = likelihoods[1]
        categorical = likelihoods[2]
        discrete.W_d.requires_grad = set
        discrete.mu_d.requires_grad = set
        categorical.mu_d.requires_grad = set
        categorical.mu_d.requires_grad = set

def fix_variational_params(model,set = False):

    model.q_s_param.requires_grad = set
    model.q_log_var.requires_grad = set
    model.q_z_mean.requires_grad = set

def BIC_criteria(ll_k, N, K, Dc, Dd,Db, L):

    dof = K*(1+(2*Dc)+Dd+(Dc*L)+(Dd*L)+(Db*L))
    bic_cost = (-2*ll_k) + (np.log(N)*dof)
    return bic_cost

def data_imputation(bin,real,dat_type,z_mean,W_c,W_d,W_b,mu_b,mu_d,var_c,k):

    #bin, cat, real = read_dataset.get_eb2_data()


    list_range = range(0, len(bin))
    missing_idx = random.choices(list_range, k=20)

    hat_samples=[]
    real_samples=[]

    if dat_type=='cat':
        for i in missing_idx:
            eta_softmax = torch.matmul(W_d[k, :, :].double(), z_mean[i, :].double().t()).t() +mu_d[:, k].double() # de mu_d quitar 0 poner K
            print(eta_softmax)
            p = distributions.categorical.Categorical(eta_softmax.t())
            hat_sample = p.sample()
            true_sample = cat[i]
            hat_samples.append(hat_sample)
            real_samples.append(true_sample)


    if dat_type =='bin':
        eta_list = []
        for i in missing_idx:
            #est_bin_params = softmax(np.dot(W_b[k, :, :].detach().numpy(),z_mean[i, :].t().detach().numpy()) + mu_b[:, k].detach().numpy())
            #hat_sample = nodes.Bernoulli(est_bin_params).random()[0]

            eta_softmax = torch.nn.functional.softmax(torch.matmul(W_b[k, :, :].double(), z_mean[i, :].double().t()) + mu_b[:, k].double(), dim=0)

            p = distributions.bernoulli.Bernoulli(eta_softmax)
            hat_sample = p.sample()

            true_sample = bin[i]
            hat_samples.append(hat_sample.numpy())
            real_samples.append(true_sample)
            eta_list.append(eta_softmax.detach().numpy())
        print(f'Hat_samples:{hat_samples}')
        print(f'True samples:{real_samples}')
        return hat_samples, real_samples, eta_list


    if dat_type == 'real':

        for i in missing_idx:
            Dc = W_c[k, :, :].shape[0]
            est_mean = np.dot(W_c[k, :, :].detach().numpy(),z_mean[i, :].t().detach().numpy())
            est_var = np.linalg.inv(var_c[k,:].detach().numpy()*np.eye(Dc))
            hat_sample = nodes.Gaussian(est_mean,est_var).random()
            true_sample = real[i]
            hat_samples.append(hat_sample)
            real_samples.append(true_sample)
        print(f'Hat_samples:{hat_samples}')
        print(f'True samples:{real_samples}')
        return hat_samples, real_samples


def data_imputation_mixture(bin,real,dat_type,mu_c,var_c,mu_d,k):
    list_range = range(0, len(bin))
    missing_idx = random.choices(list_range, k=20)

    hat_samples = []
    real_samples = []


    if dat_type == 'bin':
        for i in missing_idx:
            hat_sample = nodes.Bernoulli(mu_d[:,k]).random()
            true_sample = bin[i]
            hat_samples.append(hat_sample)
            real_samples.append(true_sample)
        print(f'Hat_samples:{hat_samples}')
        print(f'True samples:{real_samples}')
        return hat_samples, real_samples

    if dat_type == 'real':

        for i in missing_idx:
            Dc = var_c.shape[0]
            var = var_c[:,k]* np.eye(Dc)
            hat_sample = nodes.Gaussian(mu_c[:,k], var).random()
            true_sample = real[i]
            hat_samples.append(hat_sample)
            real_samples.append(true_sample)
        print(f'Hat_samples:{hat_samples}')
        print(f'True samples:{real_samples}')
        return hat_samples, real_samples


def calculate_error(dat_type,hat_samples,true_samples,probs=None):

    N = len(hat_samples)

    #hat_samples_arr= np.asarray(hat_samples)

    #true_samples_arr = np.asarray(true_samples)
    acc=0
    if dat_type=='cat':
        error = 1- metrics.accuracy_score(true_samples,hat_samples)
    elif dat_type=='bin':
        for i in range(len(true_samples)):
            acc+=(1- metrics.accuracy_score(true_samples[i],hat_samples[i]))
        acc = acc/len(true_samples)
        #auc = metrics.roc_auc_score(true_samples,probs)
        return acc
    elif dat_type=='real':
        error = metrics.mean_squared_error(true_samples,hat_samples)
        return error



import toy_distributions as dist
import util
import torch
import numpy as np
from model import Mixture_Model
import model
import test_model_K
from likelihoods.gaussian import Gaussian
from likelihoods.multinomial import Multinomial
from likelihoods.bernoulli import Bernoulli
from likelihoods.categorical import Categorical
from torch import autograd
import read_dataset
import matplotlib.pyplot as plt
import bayespy.plot as bpplt
from sklearn.preprocessing import StandardScaler

# []

N = 80
Dc = 5
Dd = 4
L = 5
K = 3

#X_c,X_b,s = dist.generate_mixture(K,N,Dc,Dd)
#transformer_c = StandardScaler().fit(X_c)   #standard normalization for real data
#transformer_c.transform(X_c)

X_b,X_d,X_c = read_dataset.get_eb2_data()
X_c = np.log(X_c+1)
X_b = X_b.reshape((len(X_b),1))
X_d = X_d.reshape((len(X_d),1))

X_d=util.onehot(X_d,int(np.max(X_d))) #one_hot encoding of categorical data
transformer_c = StandardScaler().fit(X_c)   #standard normalization for real data
transformer_c.transform(X_c)

# ----------   Diagonal Gaussian toy data  -----------

#X_c = dist.toy_gaussian_mixture(N,Dc,K)  # N x D

# ----------   Diagonal Gaussian toy data  -----------
'''
X_c = read_dataset.get_seizure()
X_c = np.log(X_c)+1


# ----------   Categorical toy data  -----------

#X_d = dist.toy_categorical_mixture(N,Dd,K)
#X_d = util.onehot(X_discrete,M) # N x D x (M+1)

# ----------   Categorical real data  -----------
X_d = read_dataset.get_cervical_data()
M = np.max(X_d)
X_d = util.onehot(X_d,M) # N x D x (M+1)

# ----------   Bernoulli toy data  -----------

#X_d = dist.toy_bernoulli_mixture(N,Dd,K)


# ----------   Mixture real data  -----------
binary,cat,real = read_dataset.get_hcv()

X_b = binary.reshape((len(binary),1))
#X_d = cat.reshape((len(cat),1))
X_d = cat
M = np.max(X_d)
X_d = util.onehot(X_d,M) # N x D x (M+1)
X_c = (real - np.mean(real)) / np.std(real)
'''

N = X_c.shape[0]
Dc = X_c.shape[1]
Dd = X_d.shape[1]
#Dd = 0
Db = X_b.shape[1]

K_list = [5]
L_list = [3]
dof_cost = []
profiles_list = []

ll_list_k_z =[]
error_list_k_z_real =[]
error_list_k_z_bin =[]
error_list_k_z_bin2 =[]
i=0
for K in K_list:
    print(f'-------  K = {K}  --------')

    for L in L_list:
        print(f'------- L = {L}  --------')

        #  ----------- Model ------------
        ll_k,z_mean,W_c,W_b,mu_b,mu_d,W_d,var_c,profiles= test_model_K.test_model(N,Dc,Dd,Db,L,K,X_c=X_c,X_d=X_d,X_b=X_b)
        #print(profiles)
        dof_k = util.BIC_criteria(ll_k, N, K, Dc, Dd,Db, L)
        dof_cost.append(dof_k)
        profiles_list.append(profiles)
        ll_list_k_z.append(ll_k)

        dat_type = 'real'
        hat_samples, real_samples = util.data_imputation(X_b, X_c, dat_type, z_mean, W_c, W_d, W_b, mu_b, mu_d, var_c,i)
        rmse = util.calculate_error(dat_type, transformer_c.inverse_transform(hat_samples), real_samples)
        error_list_k_z_real.append(rmse)
        print(f'RMSE_{K}_L{L}: {rmse}')


        dat_type = 'bin'
        hat_samples2, real_samples2,eta = util.data_imputation(X_b, X_c, dat_type, z_mean, W_c, W_d, W_b, mu_b, mu_d, var_c,i)
        acc = util.calculate_error(dat_type, hat_samples2, real_samples2,probs=eta)
        error_list_k_z_bin.append(acc)
        print(f'ACC_{K}_L{L}: {acc}')
    i+=1

result_LL = np.asarray(ll_list_k_z)
result_error_bin = np.asarray(error_list_k_z_bin)
result_error_real = np.asarray(error_list_k_z_real)
np.savetxt("results_LL.csv",result_LL , delimiter=",")
np.savetxt("results_ERROR_AUC.csv",result_error_bin , delimiter=",")
np.savetxt("results_ERROR_ACC.csv",error_list_k_z_bin , delimiter=",")
np.savetxt("results_ERROR_real.csv",result_error_real , delimiter=",")
print(W_c)
print(W_d)
#print(dof_cost)
idx_opt = np.argmin(dof_cost)
K_optimo = K_list[idx_opt]
print(' K optimo: ')
print(K_optimo)

profiles_est = profiles_list[idx_opt]
print(profiles_est)
plt.figure()
plt.plot(profiles_est,'*')
plt.legend(['Estimated s'])
plt.savefig('Profiles_comparison.png')

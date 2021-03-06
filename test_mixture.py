
import numpy as np
import read_dataset
import util
from sklearn.preprocessing import StandardScaler
from behaviorcpd import util as ut
from behaviorcpd.likelihoods.gaussian import SphereGaussian, DiagonalGaussian, KernelGaussian
from behaviorcpd.likelihoods.bernoulli import Bernoulli
from behaviorcpd.likelihoods.categorical import Categorical
# []

X_b,X_d,X_c = read_dataset.get_eb2_data()
X_c = np.log(X_c+1)
X_b = X_b.reshape((len(X_b),1))
X_d = X_d.reshape((len(X_d),1))

X_d=util.onehot(X_d,int(np.max(X_d))) #one_hot encoding of categorical data
transformer_c = StandardScaler().fit(X_c)   #standard normalization for real data
transformer_c.transform(X_c)

Dc = X_c.shape[1]
Dd = X_d.shape[1]
#Dd = 0
Db = X_b.shape[1]

Y= [X_c,X_b]
K=5
likelihood_list = [DiagonalGaussian(Dc,K), Bernoulli(Db,K)]#, KernelGaussian(T3,K)]

k, model= ut.best_model(Y, likelihood_list, max_k=5, criteria='bic', plot_s=False, plot_i=False)

print(k)
mu_c = model.likelihood_list[0].mu
var_c = model.likelihood_list[0].var
mu_d = model.likelihood_list[1].mu

dat_type = 'bin'
hat,true = util.data_imputation_mixture(X_b,X_c,dat_type,mu_c,var_c,mu_d,k)
acc = util.calculate_error(dat_type, hat, true,probs=mu_d)
print(f'ACC_{k}: {acc}')
dat_type = 'real'
hat,true = util.data_imputation_mixture(X_b,X_c,dat_type,mu_c,var_c,mu_d,k)
rmse = util.calculate_error(dat_type, transformer_c.inverse_transform(hat), true)
print(f'RMSE_{k}: {rmse}')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, scale
from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import stable_cumsum
import sklearn.linear_model as lm
import src.random_matrix as rm
import statsmodels.api as sm
from scipy import stats
import math

class LinearFactorModel:
    def __init__(self):
        self.models = None
        self.factors = None
        self.dr = None

    def fit_ols(self, fr, data, intercept=False):
        self.models = lm.LinearRegression(fit_intercept=intercept).fit(fr, data)

    def compute_residuals(self, window, fr):
        raise NotImplementedError

    def estimate_factors(self, data):
        raise NotImplementedError

class Pca(LinearFactorModel):
    dr_id = 'pca'
    def __init__(self, pct_var=None, rmt=None, corr=False, dr_id='pca'):
        self.dr_id = dr_id
        # Set the function that select the number of components to include in the model.
        self.set_k_select_fn(pct_var, rmt)
        self.setup_pca(corr)
        self.fit_id = None
        self.factors = None

    def setup_pca(self, corr):
        if corr:
            self.dr = Pipeline([('normalize', StandardScaler()), ('pca', PCA(svd_solver='full'))])
            self.dr_id = 'Corr_' + self.dr_id
        else:
            self.dr = Pipeline([('pca', PCA(svd_solver='full'))])

    def set_k_select_fn(self, pct_var, rmt):
        if pct_var is None:
            if rmt is None:
                raise ValueError('Must set pct_var or rmt variables')
            else:
                self.select_k_comp = self.mp_filter
                self.id='{}|Rmt'.format(self.dr_id)
        else:
            self.select_k_comp = self.select_pct_var
            self.pct_var = pct_var
            self.id = '{}|PctVar={}'.format(self.dr_id, pct_var)

    def estimate_factors(self, data):
        cent = pd.DataFrame(scale(data, with_mean=True, with_std=False), 
            index=data.index, columns=data.columns.values)
        self.dr.fit(cent.values)

    def project_factors(self, data):
        # x = self.dr['pca'].transform(data.values)
        if self.factors is None:
            self.factors = self.select_k_comp(self.dr['pca'].components_, data)
        return np.matmul(data.values, self.factors)

    def get_n_comp(self):
        return self.n_components_

    def get_factors(self, data):
        return self.factors

    def select_pct_var(self, factors, data):
        ratio_cumsum = stable_cumsum(self.dr['pca'].explained_variance_ratio_)
        n_components = np.searchsorted(ratio_cumsum, (self.pct_var/100.0)) + 1
        self.n_components_ = n_components
        return self.dr['pca'].components_[:n_components].T
    
    def mp_filter(self, factors, data, below=False):
        gamma = data.shape[1] / data.shape[0]
        lower, upper = rm.mp_eval_bounds(gamma)
        evals = self.dr['pca'].explained_variance_
        if below:
            filtered = self.dr['pca'].components_[np.logical_or(evals > upper, evals < lower)]
        else:
            filtered = self.dr['pca'].components_[evals > upper]
        self.n_components_ = len(filtered) + 1
        return filtered.T

    def compute_residuals(self, data):
        fr = self.project_factors(data)
        self.fit_ols(fr, data)
        return data - self.models.predict(fr)

class SPca(Pca):
    dr_id = 'SPca'
    def __init__(self, k=None, alpha=None, corr=False):
        # super().__init__()
        self.dr_id = 'SPca'
        self.id = 'SPca'
        self.setup_pca(corr)
        self.subset_cols = None
        self.setup_subset_method(k, alpha)

    def setup_subset_method(self, k, alpha):
        if k is None:
            if alpha is None:
                raise ValueError('k or alpha must be set')
            else:
                self.alpha = alpha
                self.alpha_fr = 1 + alpha
                self.subset_data = self.subset_by_alpha
                self.dr_id += '_alpha={}'.format(alpha)
                self.id += '|alpha={}'.format(alpha)
        else:
            self.subset_data = self.subset_by_k
            self.dr_id += '_k={}'.format(k)
            self.id += '|k={}'.format(k)
            self.k = k

    # def get_factors(self, data):
    #     return self.factors.T
    #     # return np.array([thresh_vec(comps[:, i]) for i in range(comps.shape[1])]).T

    def project_factors(self, data):
        return np.matmul(data.loc[:, self.subset_cols].values, self.factors)
        
    def subset_by_alpha(self, data):
        var = data.var()
        noise_thr = var.median() * self.alpha_fr
        # print(var[var > noise_thr].index.values)
        self.subset_cols = var[var > noise_thr].index.values
        self.n_components_ = len(self.subset_cols)
        return data[self.subset_cols]

    def subset_by_k(self, data):
        self.subset_cols = data.var().sort_values()[-self.k:].index.values
        return data[self.subset_cols]
        
    def estimate_factors(self, data, k=61):
        # Subset data
        self.n_components_ = k
        # self.subset_cols = data.var().sort_values()[-k:].index.values
        cent = pd.DataFrame(scale(data, with_mean=True, with_std=False), 
            index=data.index, columns=data.columns.values)

        subset = self.subset_data(cent)
        self.dr.fit(subset)
        comps = self.dr['pca'].components_
        # Scaling factor: sqrt(2logk)
        sf = np.sqrt(2 * np.log(comps.shape[0]))
        self.factors = np.apply_along_axis(SPca.thresh_vec, 1, comps, sf).T

    @staticmethod
    def thresh_vec(factor, sf):
        # tau = stats.median_absolute_deviation(factor) / .6745
        tau = np.var(factor)
        delta = tau * sf
        factor[np.abs(factor) < delta] = 0
        return factor

class RPca(LinearFactorModel):
    def __init__(self, mu=None, lmbda=None):
        self.config = False
        self.dr_id = 'RPca'
        self.id = 'RPca'
        self.dr = self
        # if mu:
        # else:
        #     self.mu = 


        # if lmbda:
        #     self.lmbda = lmbda
        # else:

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def setup(self, D):
        self.S = np.zeros(D.shape)
        self.Y = np.zeros(D.shape)

    def estimate_factors(self, data):
        if not self.config:
            self.setup(data)
            self.config = True
        self.fit(data.values)

    def compute_residuals(self, data):
        return data - self.L

    def get_n_comp(self):
        return np.linalg.matrix_rank(self.L)

    def fit(self, D, tol=None, max_iter=1000, iter_print=100, verbose=False):
        self.mu = np.prod(D.shape) / (4 * self.frobenius_norm(D))
        self.mu_inv = 1 / self.mu
        self.lmbda = 1 / np.sqrt(np.max(D.shape))
        it = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(D)

        while (err > _tol) and it < max_iter:
            Lk = self.svd_threshold(D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (D - Lk - Sk)
            err = self.frobenius_norm(D - Lk - Sk)
            it += 1
            if verbose:
                if (it % iter_print) == 0 or it == 1 or it > max_iter or err <= _tol:
                    print('iteration: {0}, error: {1}'.format(it, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')

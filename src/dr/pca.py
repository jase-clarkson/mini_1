import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, scale
from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import stable_cumsum
import src.random_matrix as rm
from .lfm import LinearFactorModel

class Pca(LinearFactorModel):
    dr_id = 'pca'
    def __init__(self, pr=None, rmt=None, corr=False, dr_id='pca'):
        self.dr_id = dr_id
        # Set the function that select the number of components to include in the model.
        self.set_k_select_fn(pr, rmt)
        self.setup_pca(corr)
        self.fit_id = None
        self.factors = None

    def setup_pca(self, corr):
        if corr:
            self.dr = Pipeline([('normalize', StandardScaler()), ('pca', PCA(svd_solver='full'))])
            self.dr_id = 'Corr_' + self.dr_id
        else:
            self.dr = Pipeline([('pca', PCA(svd_solver='full'))])

    def set_k_select_fn(self, pr, rmt):
        if pr is None:
            if rmt is None:
                raise ValueError('Must set pr or rmt variables')
            else:
                self.select_k_comp = self.mp_filter
                self.id='{}|Rmt'.format(self.dr_id)
        else:
            self.select_k_comp = self.select_pr
            self.pr = pr
            self.id = '{}|Pr={}'.format(self.dr_id, pr)

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

    def select_pr(self, factors, data):
        ratio_cumsum = stable_cumsum(self.dr['pca'].explained_variance_ratio_)
        n_components = np.searchsorted(ratio_cumsum, (self.pr/100.0)) + 1
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
        if len(filtered) == 0:
            filtered = self.dr['pca'].components_[0]
        self.n_components_ = len(filtered) + 1
        return filtered.T

    def compute_residuals(self, data):
        fr = self.project_factors(data)
        self.fit_ols(fr, data)
        return data - self.models.predict(fr)

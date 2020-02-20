import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

    def project_factors(self, data):
        raise NotImplementedError

    def fit_ols(self, fr, data):
        self.models = lm.LinearRegression(fit_intercept=False).fit(fr, data)

    def compute_residuals(self, window, fr):
        return window - self.models.predict(fr)

    def estimate_factors(self, data):
        self.dr.fit(data)

class Pca(LinearFactorModel):
    def __init__(self, pct_var=None, rmt=None, dr_id='Pca'):
        if pct_var is None:
            if rmt is None:
                raise ValueError('Must set pct_var or rmt variables')
            else:
                self.select_k_comp = self.mp_filter
                self.id='{}|Rmt'.format(dr_id)
        else:
            self.select_k_comp = self.select_pct_var
            self.pct_var = pct_var
            self.id = '{}|PctVar={}'.format(dr_id, pct_var)


        # self.dr = Pipeline([('normalize', StandardScaler()), ('pca', PCA(svd_solver='full'))])
        self.dr = PCA()
        self.dr_id = dr_id
        self.fit_id = None

    def project_factors(self, data):
        x = self.dr.transform(data.values)
        return x

    def get_n_comp(self):
        return self.n_components_

    def get_factors(self):
        # return self.select_k_comp(self.dr['pca'].components_)
        return self.select_k_comp(self.dr.components_)

    def select_pct_var(self, factors):
        ratio_cumsum = stable_cumsum(self.dr.explained_variance_ratio_)
        n_components = np.searchsorted(ratio_cumsum, (self.pct_var/100.0)) + 1
        self.n_components_ = n_components
        return self.dr.components_[:n_components]
    
    def mp_filter(self, factors):
        pass

class SPca(Pca):
    def __init__(self, pct_var=None, rmt=None):
        super().__init__(pct_var=pct_var, rmt=rmt, dr_id='SPca')

    def get_factors(self):
        # Threshold-factor pairs
        comps = self.select_k_comp(self.dr.components_)
        return np.array([thresh_vec(factor) for factor in comps])

def thresh_vec(factor):
    th = stats.median_absolute_deviation(factor) / .6745
    factor[np.abs(factor) < th] = 0
    return factor


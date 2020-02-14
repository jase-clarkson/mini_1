import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.linear_model as lm
import src.random_matrix as rm
import statsmodels.api as sm
import math

class LinearFactorModel:
    def __init__(self):
        self.models = None
        self.factors = None
        self.fr = None
    
    def estimate_factors(self, data):
        raise NotImplementedError

    def project_factors(self, data):
        raise NotImplementedError

    @staticmethod
    def estimate_betas_ols(data, fr):
        """ Fit OLS models for the factors the the data"""
        # return data.apply(lambda x: lm.LinearRegression(fit_intercept=False).fit(fr, data[x.name]))
        return lm.LinearRegression(fit_intercept=False).fit(fr, data)

    def estimate_fm_ols(self, data):
        self.fr = self.estimate_factors(data)
        self.models = LinearFactorModel.estimate_betas_ols(data, self.fr)
        return self.compute_residuals(data, self.fr)

    def compute_residuals(self, data, fr=None):
        if fr is None:
            fr = self.project_factors(data)
        return data - self.models.predict(fr)

class PcaPctVar(LinearFactorModel):
    def __init__(self, data, pct_var=50):
        self.scores = pd.DataFrame(columns=data.columns.values, index=data.index)
        self.pca_info = pd.DataFrame(index=data.index, columns=['n_comp'])
        self.pct_var = pct_var
        self.pca = PCA(n_components=(pct_var/100.0), svd_solver='full')
        self.scaler = StandardScaler()
        self.corr_pca = Pipeline([('normalize', self.scaler), ('pca', self.pca)])
        
    def estimate_factors(self, data):
        """ 
        Estimate factors from data using PCA, choosing the number of components such that at least 
        (pct)% of the total explainable variance is retained. Return the data projected onto new subspace.
        Notice that we perform PCA on the correlation matrix, not just the covariance matrix. This is 
        equvialent to performing PCA on the covariance matrix of standardized (zero mean, unit variance)
        data. 
        """
        fr = self.corr_pca.fit_transform(data)
        date = data.iloc[-1].name
        self.pca_info.loc[date, 'n_comp'] = self.pca.n_components_
        return fr

    def project_factors(self, data):
        return self.corr_pca.transform(data.values)

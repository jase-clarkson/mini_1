from pca import Pca
import numpy as np
import pandas as pd
from scipy import stats


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
    def thresh_vec(factor, sf, robust=True):
        if robust:
            tau = stats.median_absolute_deviation(factor) / .6745    
            delta = tau * sf
        else:
            delta = np.var(factor)
        factor[np.abs(factor) < delta] = 0
        return factor

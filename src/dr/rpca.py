from lfm import LinearFactorModel
import numpy as np
import pandas as pd

# Code adapted from from https://github.com/dganguli/robust-pca
class RPca(LinearFactorModel):
    def __init__(self, mu=None, lmbda=None):
        self.config = False
        self.dr_id = 'RPca'
        self.id = 'RPca'
        self.dr = self

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

#     def plot_fit(self, size=None, tol=0.1, axis_on=True):

#         n, d = self.D.shape

#         if size:
#             nrows, ncols = size
#         else:
#             sq = np.ceil(np.sqrt(n))
#             nrows = int(sq)
#             ncols = int(sq)

#         ymin = np.nanmin(self.D)
#         ymax = np.nanmax(self.D)
#         print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

#         numplots = np.min([n, nrows * ncols])
#         plt.figure()

#         for n in range(numplots):
#             plt.subplot(nrows, ncols, n + 1)
#             plt.ylim((ymin - tol, ymax + tol))
#             plt.plot(self.L[n, :] + self.S[n, :], 'r')
#             plt.plot(self.L[n, :], 'b')
#             if not axis_on:
#                 plt.axis('off')

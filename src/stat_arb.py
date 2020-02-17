import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
import src.random_matrix as rm
import statsmodels.api as sm
import math

class Strategy:
    def __init__(self):
        self.cal_freq = None

    def init(self, data):
        raise NotImplementedError

    def set_calibration_dates(self, data):
        if self.cal_freq is None:
            raise NotImplementedError('Must set cal_freq in constructor of {}'.format(self.__class__.__name__))
        # TODO: make cal freq more general than days and figure out how to divide
        # into ranges like [date 1, ..., date n] / cal_freq.
        n_recal = math.ceil(len(data) / float(self.cal_freq))
        cal_ix = [i * self.cal_freq for i in range(n_recal)]
        self.cal_dates = data.iloc[cal_ix].index

class StArbFm:
    def __init__(self, data, fm, alpha=0.9, cal_freq=60, window_len=60, forecast_horizon=1):
        # alpha is ewm parameter
        self.alpha = alpha
        self.fm = fm

        self.cal_freq = cal_freq
        self.window_len = window_len
        self.forecast_horizon = 1

        self.portfolio = None
        self.portfolios = None
        self.pnl = None
        self.pca_info = None

    def init(self, data):
        self.portfolios = pd.DataFrame(columns=data.columns.values)
        self.pnl = pd.DataFrame(columns=['PnL'])
        # TODO: below should contain Mihai scoring function
        # self.scores = pd.DataFrame(columns=data.columns.values, index=data.index)
        self.pca_info = pd.DataFrame(index=data.index, columns=['n_comp'])
        # TODO: add a check that this has been called before backtesting.

    def calibrate_portfolio(self, data, date):
        """ Calibrate the portfolio (from T-N,...,T-1) and return weights for time T"""        
        if date in self.cal_dates:
            ix = data.index.get_loc(date)
            if len(data) - ix < self.forecast_horizon or ix < self.window_len:
                return
            window = data.iloc[ix-self.window_len:ix]
            res = self.fm.estimate_fm_ols(window)
            # TODO: refactor into more general 'tracker' object.
            self.pca_info.loc[date, 'n_comp'] = self.fm.n_components_
            # From the residuals, compute portfolio weights.
            self.portfolio = self.compute_portfolio_weights(res)


    def update(self, day_returns, date):
        self.pnl.loc[date] = np.dot(self.portfolio, day_returns)
        # Record the portfolio weights on that day
        self.portfolios.loc[date, :] = self.portfolio

    # TODO: make this generic to allow other pos sizing strategies.
    def compute_portfolio_weights(self, res):
        return res.ewm(span=5).mean().iloc[-1]

def score_sign(signal, data):
    return np.sum(data * np.sign(signal))

def compute_vn_factor_return(factor, stock_returns, stdevs):
    """ Compute vol-normalized factor returns """
    return factor.dot(stock_returns.values / stdevs)

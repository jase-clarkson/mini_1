import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from sklean.Pipeline import Pipeline
import sklearn.linear_model as lm
from sklearn.preprocessing import scale
import src.random_matrix as rm
import statsmodels.api as sm
import math

class Strategy:
    def __init__(self, window_len=60):
        self.cal_freq = None
        self.id = ''
        self.window_len = window_len

    def setup(self, data):#, transformers):
        """ 
        Performs backtest-level configuration tasks for the given strategy. Initialise
        relevant objects (some depend on the data, some need to be reset to null case before
        each run
        """
        self.set_calibration_dates(data)
        # Used to store current portfolio.
        self.portfolio = None
        # Record the portfolio weights on each day.
        self.portfolios = np.empty(data.shape)
        # Record days on which the strategy traded/rebalanced.
        self.trading_days = np.empty(data.shape[0], dtype=pd.Timestamp)
        # Tracks array index of current portfolio
        self.pf_ix = 0
        # Record number of stocks used in session.
        self.n_stocks = data.shape[1]
        # Perform non-generic bt-level setup.
        self.init(data)
    
    def set_calibration_dates(self, data):
        if self.cal_freq is None:
            raise NotImplementedError('Must set cal_freq in constructor of {}'.format(self.__class__.__name__))
        # TODO: make cal freq more general than days and figure out how to divide
        # into ranges like [date 1, ..., date n] / cal_freq.
        n_recal = math.ceil(len(data) / float(self.cal_freq))
        cal_ix = [i * self.cal_freq for i in range(1, n_recal)]
        self.cal_dates = data.iloc[cal_ix].index

    def compute_log_returns(self, raw_lr):
        """ The idea of this is one can apply vol norm or whatever they want after."""
        return raw_lr

    def pnl_reconciliation(self, data, filter_outliers=True):
        # Cleanup array        
        self.trading_days = self.trading_days[~pd.isna(self.trading_days)]
        # Remove null days
        self.portfolios = self.portfolios[:self.pf_ix, :]
        # Create dataframe from portfolio weights for easier manipulation.
        self.portfolios = pd.DataFrame(self.portfolios, index=self.trading_days, columns=data.columns.values)
        self.portfolios = self.portfolios #* self.portfolios.expanding().std().fillna(value=1) / 2
        # Perform row-wise inner products between pf weights and daily returns.
        # This is achieved by Hadamard producit and then row-wise sum.
        self.raw_lr = pd.DataFrame(
            np.sum(data.loc[self.trading_days, :].values * self.portfolios, axis=1),
            index=self.trading_days, columns=[self.id])
        if filter_outliers:
            self.raw_lr = self.raw_lr[self.raw_lr < (self.raw_lr.std() * 5)]
        # The compute_log_returns custom function allows one to apply ex-post adjustments
        # that reflect sizing, for example rolling vol-normalization.
        self.lr = self.compute_log_returns(self.raw_lr)
        # Compute the pnl process.
        self.cum_ret = self.lr.cumsum()

    def init(self, data):
        """Custom setup function for individual strategies if they need any backtest-time config based on the data."""
        raise NotImplementedError()
    
    def update(self, date, transforms):
        if self.portfolio is not None:
            # self.portfolios[self.pf_ix] = np.dot(self.portfolio.values, day_returns.values)
            self.portfolios[self.pf_ix, :] = self.portfolio.values
            self.trading_days[self.pf_ix] = date
            self.update_trackers(self.pf_ix)
            self.pf_ix += 1

    def update_trackers(self, pf_ix):
        return

    def register_transforms(self, tf_container):
        """
        Before each backtest, one must register the transforms they want to be applied
        to the window data passed to the calibrate_portfolio function. This is to avoid
        repeated calculations for strategies using the same transforms.
        """
        return NotImplementedError
        
class StArbFm(Strategy):
    def __init__(self, fm, span=5, cal_freq=60, window_len=60, forecast_horizon=1):
        super().__init__()
        # alpha is ewm parameter
        self.span = span
        self.fm = fm

        self.cal_freq = cal_freq
        self.window_len = window_len
        self.forecast_horizon = 1

        self.portfolio = None
        # self.portfolios = None
        self.returns = None
        self.pca_info = None

        self.id = fm.id + '|sp={}|cal={}|win={}'.format(span, cal_freq, window_len)

    def init(self, data):
        self.n_pcs = np.empty(data.shape[0])
        # TODO: add a check that this has been called before backtesting.

    def calibrate_portfolio(self, data, date, transforms=None):
        """ Calibrate the portfolio (from T-N,...,T-1) and return weights for time T"""        
        if date in self.cal_dates:
            ix = data.index.get_loc(date)
            if len(data) - ix < self.forecast_horizon or ix < self.window_len:
                return
            # NOTE: iloc slice doesn't include ix (so no lookahead).
            window = data.iloc[ix-self.window_len:ix]
            # TODO: could have a rolling 60-day mean precomputed for each col before?
            # TODO: each strat can have it's own set of precomputed data it can access.
            # win_cent = pd.DataFrame(scale(window, with_mean=True, with_std=False), 
            # index=window.index, columns=window.columns.values)
            self.fm.factors = None
            # If the factors have already been calculated for this window and just need to choose k diff.
            if self.id_tf_map[self.transform_id] != date:
                # Fit to new window data
                self.fm.estimate_factors(window)                
                self.id_tf_map[self.transform_id] = date
            # factors = self.fm.get_factors(win_cent)
            # Compute factor returns
            # fr = np.matmul(win_cent.values, factors.T)
            res = self.fm.compute_residuals(window)
            # From the residuals, compute portfolio weights.
            self.portfolio = self.compute_portfolio_weights(res)
        self.update(date, transforms)

    # TODO: rename this to apply bet sizing or something
    def compute_log_returns(self, raw_returns):
        return raw_returns / raw_returns.ewm(span=252).var().shift(1)

    # TODO: make this generic to allow other pos sizing strategies.
    def compute_portfolio_weights(self, res):
        return -1 * res.ewm(span=self.span).mean().iloc[-1]

    def update_trackers(self, pf_ix):
        self.n_pcs[pf_ix] = self.fm.get_n_comp()
    
    def register_transforms(self, tf_container):
        self.id_tf_map = tf_container.id_tf_map
        self.transform_id = self.fm.dr_id + '_{}'.format(self.window_len)
        if not tf_container.register(self.fm, self.transform_id):
            # If two are using the same underlying transform, have them point to same object
            self.fm.dr = tf_container.get_dr(self.transform_id)

class Transforms:
    def __init__(self):
        self.tfmrs = {}
        self.id_tf_map = {}

    def register(self, tf, t_id):
        if t_id not in self.tfmrs:
            self.tfmrs[t_id] = tf.dr
            self.id_tf_map[t_id] = ''
            print('Registered {}'.format(t_id))
            return True
        return False
    
    def get_dr(self, str):
        return self.tfmrs[str]

def score_sign(signal, data):
    return np.sum(data * np.sign(signal))

def compute_vn_factor_return(factor, stock_returns, stdevs):
    """ Compute vol-normalized factor returns """
    return factor.dot(stock_returns.values / stdevs)

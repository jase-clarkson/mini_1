import numpy as np
import pandas as pd

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
        n_recal = int(np.ceil(len(data) / float(self.cal_freq)))
        cal_ix = [i * self.cal_freq for i in range(1, n_recal)]
        self.cal_dates = data.iloc[cal_ix].index

    def apply_pos_sizing(self, raw_lr):
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
        # self.rebalance_days = self.cal_dates & self.trading_days
        self.raw_lr = pd.DataFrame(
            np.sum(data.loc[self.trading_days, :].values * self.portfolios.loc[self.trading_days, :], axis=1),
            index=self.trading_days, columns=[self.id])
        if filter_outliers:
            self.raw_lr = self.raw_lr[self.raw_lr < (self.raw_lr.std() * 5)]
        # The compute_log_returns custom function allows one to apply ex-post adjustments
        # that reflect sizing, for example rolling vol-normalization.
        self.lr = self.apply_pos_sizing(self.raw_lr)
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

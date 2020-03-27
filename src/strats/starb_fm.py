from .strategy import Strategy
import numpy as np
import pandas as pd

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
            self.fm.factors = None
            # If the factors have already been calculated for this window and just need to choose k diff.
            if self.id_tf_map[self.transform_id] != date:
                # Fit to new window data
                self.fm.estimate_factors(window)                
                self.id_tf_map[self.transform_id] = date
            res = self.fm.compute_residuals(window)
            # From the residuals, compute portfolio weights.
            self.portfolio = self.compute_portfolio_weights(res)
        self.update(date, transforms)

    def apply_pos_sizing(self, raw_returns):
#         return raw_returns / raw_returns.ewm(span=252).std().shift(1)
        return self.size_rolling_kelly(raw_returns, 1)
    
    def size_rolling_kelly(self, returns, frac=0.5):
        return frac * returns / returns.ewm(span=252).std().shift(1)

    # TODO: make this generic to allow other pos sizing strategies.
    def compute_portfolio_weights(self, res):
        # TODO: do we need to normalize this to be optimal??
        return -1 * res.ewm(span=self.span).mean().iloc[-1]

    def update_trackers(self, pf_ix):
        self.n_pcs[pf_ix] = self.fm.get_n_comp()
    
    def register_transforms(self, tf_container):
        self.id_tf_map = tf_container.id_tf_map
        self.transform_id = self.fm.dr_id + '_{}'.format(self.window_len)
        if not tf_container.register(self.fm, self.transform_id):
            # If two are using the same underlying transform, have them point to same object
            self.fm.dr = tf_container.get_dr(self.transform_id)

    def plot_dimensionality(self):
        pcs_df = pd.DataFrame(self.n_pcs[:self.pf_ix], index=self.trading_days, columns=[self.id])
        pcs_df.plot(figsize=(12, 4), c='k', legend=False, linewidth=.8)
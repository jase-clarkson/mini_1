import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from os import mkdir


def backtest(df_lr, strats, benchmark=None, it=500, savefig=True):    
    i = 0
    tfmrs = Transforms()
    for strat in strats:
        strat.register_transforms(tfmrs)
        strat.setup(df_lr)
    # TODO: consider using itertuples (supposedly faster, less clear but maybe worth it)
    for index, row in df_lr.iterrows():
        for strat in strats:
            strat.calibrate_portfolio(df_lr, index, tfmrs)
        i += 1
        if i % it == 0 and i > 250: print('Iteration: {}'.format(i))

    for strat in strats:
        strat.pnl_reconciliation(df_lr, filter_outliers=True)
        sr_test = sharpe_ratio_test(strat.lr)
        print('{} | Sharpe: {} | p-value: {}'.format(strat.id, sr_test[0], sr_test[1]))
    plot_drawdowns(strats, benchmark, savefig=savefig)
    
def plot_drawdowns(strats, benchmark, savefig=False):
    rets = [strat.cum_ret for strat in strats]
    rets = pd.concat(rets, axis=1)
    if benchmark is not None:
        rets *= benchmark.std()
    rets.plot(figsize=(10,6))
    if benchmark is not None:
        benchmark.cumsum().plot(label='SPY returns')
    plt.legend()
    plt.grid(axis='y', alpha=.7)
    if savefig:
        try:
            mkdir('figs')
        except FileExistsError:
            pass
        plt.savefig('figs/{}.pdf'.format(pd.Timestamp.now()), format='pdf', dpi=1200, bbox_inches='tight')

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

def sharpe_ratio_test(log_ret):
    '''
    Implements the sharpe ratio test ion Opdyke 2007
    :param log_ret: daily log-returns
    :return: annualised sharpe ratio and p-value
    '''
    sharpe = log_ret.mean()/log_ret.std()
    sharpe_annualised = sharpe * np.sqrt(252)    
    T = len(log_ret)
    # std = log_ret.std()
    skew = ((log_ret - log_ret.mean()) ** 3).mean()
    kurtosis = ((log_ret - log_ret.mean()) ** 4).mean()    
    sharpe_se = np.sqrt((1 + sharpe ** 2/ 4 * (kurtosis - 1) - sharpe * skew) / T)    
    p_value = 1 - norm.cdf(sharpe/sharpe_se)    
    return [sharpe_annualised.round(2).values[0], p_value[0]]

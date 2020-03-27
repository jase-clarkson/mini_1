import numpy as np
import matplotlib.pyplot as plt


# TODO: change name of this to eigval for clarity.
def mp_eval_bounds(gamma, sigma=1):
    """
    Compute the max and min eigenvalues for the MP density parameterised by lambda=lam and sigma
    :returns maximum eigenvalue, minimum eigenvalue
    """
    return np.power(sigma * (1 - np.sqrt(gamma)), 2), np.power(sigma * (1 + np.sqrt(gamma)), 2)


def mp_pdf(x, p, n, sigma=1):
    '''
    Evaluate the Marcenko-Pastur density at x, where the MP density is parameterised by sigma and gamma=p/n and X is a pxp matrix of the form AA', where A is pxn.
    :param x: The data points x at which to evaluate the density.
    :param p: The number of features.
    :param n: The number of data points.
    :param sigma: The variance of the data generating distribution.
    '''
    gamma = p / n
    print('Gamma: {}'.format(gamma))
    # TODO: update this to include both cases of the MP-density.
#     assert Q >= 1, "Invalid dimension ratio for input matrix X"
    min_eval, max_eval = mp_eval_bounds(gamma, sigma)

    def f(z):
        return (np.sqrt((max_eval - z)*(z - min_eval))) / (2 * np.pi * (sigma ** 2) * gamma * z)
    y = np.copy(x)
#     print('Unfiltered: {}'.format(y))

    supported = (y >= min_eval) & (y <= max_eval)
#     print('Supported: {}'.format(supported))
    np.putmask(y, supported, f(y))
    np.putmask(y, ~supported, 0)
#     print('Filtered: {}'.format(y))
#     print('Number supported evals: {}'.format(sum(supported)))
    return y


def compare_spectrum_to_mp(X, n, sigma, upper=None, save=False):
    '''
    Overlay the relevant Marcenko-Pastur density over the spectrum of the matrix X.
    :param X: Matrix of the form 1/p * AA'.
    :param T: The number of columns in the matrix A (where A is an pxn matrix).
    :param sigma: The variance of the (Gaussian) distribution for the entries of the random matrix in the mp density.
    :param upper: The upper bound on eigenvalues to appear in the plot. Example: -1 would omit the largest eigenvalue from the plots, -2 the 2 largest eigenvalues etc.
    '''
    assert np.allclose(X.T, X)
    e = np.linalg.eigvalsh(X)
    e = np.clip(e, .0001, e.max() + 1)  # Clip very small eigenvalues
    print('Data eigenvalues: ', e.min(), e.max())
    _, ax = plt.subplots(1, 1)
    ax.hist(e[:upper], density=True, bins=50, edgecolor='black', linewidth=1.2, color='white')

    ax.set_autoscale_on(True)
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Probability Density')
    p = X.shape[0]
    min, max = mp_eval_bounds(p/n, sigma)
    print('Theoretical min: {} | max: {}'.format(min, max))
    x = np.linspace(min , max, 5000)
    ax.plot(x, mp_pdf(x, p, n, sigma), linewidth=2, color='r')
    if save:
        plt.savefig('figs/mp_histo.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.show()
        
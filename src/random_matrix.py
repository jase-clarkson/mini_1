import numpy as np
import matplotlib.pyplot as plt


# TODO: change name of this to eigval for clarity.
def mp_eval_bounds(Q, sigma=1):
    """
    Compute the max and min eigenvalues for the MP density parameterised by lambda=lam and sigma
    :returns maximum eigenvalue, minimum eigenvalue
    """
    return np.power(sigma * (1 - np.sqrt(1/Q)), 2), np.power(sigma * (1 + np.sqrt(1/Q)), 2)


def mp_pdf(x, M, T, sigma=1):
    '''
    Evaluate the Marcenko-Pastur density at x, where the MP density is parameterised by
    sigma and lambda=n/p and X is a matrix of the form AA'.
    :param x: The data points x at which to evaluate the density.
    :param n: The number of data points.
    :param p: The number of features.
    :param sigma: The variance of the data generating distribution.
    '''
    Q = T / M
    print('Q: {}'.format(Q))
    # TODO: update this to include both cases of the MP-density.
    assert Q >= 1, "Invalid dimension ratio for input matrix X"
    min_eval, max_eval = mp_eval_bounds(Q, sigma)

    def f(z):
        return (Q * np.sqrt((max_eval - z)*(z - min_eval))) / (z * 2 * np.pi * (sigma ** 2))
    y = np.copy(x)
#     print('Unfiltered: {}'.format(y))

    supported = (y >= min_eval) & (y <= max_eval)
#     print('Supported: {}'.format(supported))
    np.putmask(y, supported, f(y))
    np.putmask(y, ~supported, 0)
#     print('Filtered: {}'.format(y))
#     print('Number supported evals: {}'.format(sum(supported)))
    return y


def compare_spectrum_to_mp(X, T, sigma):
    '''
    Overlay the relevant Marcenko-Pastur density over the spectrum of the matrix X.
    :param X: Matrix of the form 1/T * AA'.
    :param T: The number of columns in the matrix A (where A is an MxT matrix).
    :param sigma: The variance of the (Gaussian) distribution for the entries of the random matrix in the mp density.
    '''
    assert np.allclose(X.T, X)
    e = np.linalg.eigvalsh(X)
    print(e)
    plt.plot(e[:-1])
    plt.show()
    e = np.clip(e, .0001, e.max() + 1)  # Clip very small eigenvalues
    print('True: ', e.min(), e.max())
    fig, ax = plt.subplots(1, 1)
    ax.hist(e, density=True, bins=50)
#     ax.hist(e, density=True, bins=50)

    ax.set_autoscale_on(True)
    print(X.shape)
    M = X.shape[0]
    print('M: {}'.format(M))
    min, max = mp_eval_bounds(T/M, sigma)
    print('min: {} | max: {}'.format(min, max))
    x = np.linspace(min + 0.0001, max, 5000)
    ax.plot(x, mp_pdf(x, M, T, sigma), linewidth=4, color='r')
    plt.show()

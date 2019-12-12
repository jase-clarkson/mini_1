import numpy as np
import matplotlib.pyplot as plt


def mp_eval_bounds(lam, sigma):
    """
    Compute the max and min eigenvalues for the MP density parameterised by lambda=lam and sigma
    :returns maximum eigenvalue, minimum eigenvalue
    """
    return np.power(sigma * (1 - np.sqrt(lam)), 2), np.power(sigma * (1 + np.sqrt(lam)), 2)


def mp_pdf(x, n, p, sigma):
    '''
    Evaluate the Marcenko-Pastur density at x, where the MP density is parameterised by
    sigma and lambda=n/p and X is a matrix of the form X'X.
    :param x: The data points x at which to evaluate the density
    :param n: The number of
    '''
    lam = n / p
    assert lam <= 1, "Invalid dimension ratio for input matrix X"
    min_eval, max_eval = mp_eval_bounds(lam, sigma)

    def f(z):
        return np.sqrt((max_eval - z)*(z - min_eval)) / (lam * z * 2 * np.pi * (sigma ** 2))
    y = np.copy(x)
    supported = (y >= min_eval) & (y <= max_eval)
    np.putmask(y, supported, f(y))
    np.putmask(y, ~supported, 0)
    print(y)
    print(sum(supported))
    return y


def compare_spectrum_to_mp(X, p, sigma):
    assert np.allclose(X, X.T)
    e = np.linalg.eigvalsh(X)
    print(e[:-1])
    plt.plot(e[:-1])
    plt.show()
    e = np.clip(e, .0001, e.max() + 1)  # Clip very small eigenvalues
    print('True: ', e.min(), e.max())
    fig, ax = plt.subplots(1, 1)
    ax.hist(e[:-1], density=True, bins=50)
    ax.set_autoscale_on(True)
    n = X.shape[0]
    min, max = mp_eval_bounds(n/p, sigma)
    print(min, max)
    x = np.linspace(min, max, 5000)
    ax.plot(x, mp_pdf(x, n, p, sigma), linewidth=4, color='r')
    plt.show()

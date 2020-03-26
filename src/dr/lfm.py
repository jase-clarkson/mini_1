import sklearn.linear_model as lm


class LinearFactorModel:
    def __init__(self):
        self.models = None
        self.factors = None
        self.dr = None

    def fit_ols(self, fr, data, intercept=False):
        self.models = lm.LinearRegression(fit_intercept=intercept).fit(fr, data)

    def compute_residuals(self, window, fr):
        raise NotImplementedError

    def estimate_factors(self, data):
        raise NotImplementedError

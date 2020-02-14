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
        pass

class VolNormOU:
    def __init__(self, alpha=0.9):
        # alpha is ewm parameter
        self.alpha = alpha

    def size_pos(self, res):
        weights = res.ewm(span=5).mean().iloc[-1]
        print(weights)
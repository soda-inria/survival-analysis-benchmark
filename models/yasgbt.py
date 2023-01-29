import numpy as np
from tqdm.notebook import tqdm

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor

from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator, CensoringDistributionEstimator

from models.survival_mixin import SurvivalMixin


class DataSampler:
    
    def __init__(self, y):
        self.y = y
        times, km = kaplan_meier_estimator(y["event"], y["duration"])
        # inverse_km has x: [0.01, ..., .99] and y: [0., ..., 820.]
        self.inverse_km = StepFunction(x=km[::-1], y=times[::-1])
        self.max_km, self.min_km = km[0], km[-1]
    
    def make_sample(self):
        y = self.y
        
        surv_probs = np.random.uniform(self.min_km, self.max_km, y.shape[0])
        times = self.inverse_km(surv_probs)

        mask_y_0 = y["event"] & (y["duration"] <= times)
        mask_y_1 = y["duration"] > times

        yc = np.zeros(y.shape[0])
        yc[mask_y_1] = 1
        
        cens = CensoringDistributionEstimator().fit(y)
        # calculate inverse probability of censoring weight at current time point t.
        prob_cens_t = cens.predict_proba(times)
        prob_cens_t[prob_cens_t == 0] = np.inf
        # calculate inverse probability of censoring weights at observed time point
        prob_cens_y = cens.predict_proba(y["duration"])
        prob_cens_y[prob_cens_y == 0] = np.inf

        sample_weights = np.where(mask_y_0, prob_cens_y, 0)
        sample_weights = np.where(mask_y_1, prob_cens_t, sample_weights)
                
        return times.reshape(-1, 1), yc, sample_weights


class YASGBT(BaseEstimator, SurvivalMixin):

    name = "YASGBT"

    def __init__(self, n_iter=10, learning_rate=0.01, verbose=False):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.verbose=verbose
    
    def fit(self, X, y, times=None):
        
        # TODO: add check_X_y from sksurv
        monotonic_cst = np.zeros(X.shape[1]+1)
        monotonic_cst[0] = -1

        hgbc = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            loss="squared_error",
            max_iter=1,
            warm_start=True,
            monotonic_cst=monotonic_cst,
        )

        data_sampler = DataSampler(y)
        
        iterator = range(self.n_iter)
        if self.verbose:
            iterator = tqdm(iterator)

        for _ in iterator:
            times, yc, sample_weight = data_sampler.make_sample()
            Xt = np.hstack([times, X])
            hgbc.max_iter += 1
            hgbc.fit(Xt, yc, sample_weight=sample_weight)
        
        self.hgbc_ = hgbc
        self.data_sampler_ = data_sampler
        
        return self
    
    def predict_survival_function(self, X, times):

        all_y_probs = []

        iterator = range(times)
        if self.verbose:
            iterator = tqdm(iterator)

        for t in iterator:
            t = np.full((X.shape[0], 1), t)
            Xt = np.hstack([t, X])
            y_probs = self.hgbc_.predict(Xt)
            all_y_probs.append(y_probs)
        
        return np.vstack(all_y_probs).T

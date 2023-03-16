from abc import ABC
import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from sksurv.functions import StepFunction
from sksurv.metrics import integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator, CensoringDistributionEstimator

from models.survival_mixin import SurvivalMixin
from model_selection.cross_validation import get_time_grid


class IBSTrainingSampler:
    
    def __init__(self, y, sampling_strategy, random_state):
        self.y = y
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        times, km = kaplan_meier_estimator(y["event"], y["duration"])
        # inverse_km has x: [0.01, ..., .99] and y: [0., ..., 820.]
        self.inverse_km = StepFunction(x=km[::-1], y=times[::-1])
        self.max_km, self.min_km = km[0], km[-1]
    
    def make_sample(self):
        y = self.y 
        rng = check_random_state(self.random_state)
        
        if self.sampling_strategy == "inverse_km":
            surv_probs = rng.uniform(self.min_km, self.max_km, y.shape[0])
            times = self.inverse_km(surv_probs)
        elif self.sampling_strategy == "uniform":
            min_times, max_times = y["duration"].min(), y["duration"].max()
            times = rng.uniform(min_times, max_times, y.shape[0])
        else:
            raise ValueError(f"Sampling strategy must be 'inverse_km' or 'uniform', got {self.sampling_strategy}")

        mask_y_0 = y["event"] & (y["duration"] <= times)
        mask_y_1 = y["duration"] > times

        yc = np.zeros(y.shape[0])
        yc[mask_y_1] = 1
        
        cens = CensoringDistributionEstimator().fit(y)
        # calculate inverse probability of censoring weight at current time point t.
        prob_cens_t = cens.predict_proba(times)
        prob_cens_t = np.clip(prob_cens_t, 1e-8, 1)
        # calculate inverse probability of censoring weights at observed time point
        prob_cens_y = cens.predict_proba(y["duration"])
        prob_cens_y = np.clip(prob_cens_y, 1e-8, 1)

        sample_weights = np.where(mask_y_0, 1/prob_cens_y, 0)
        sample_weights = np.where(mask_y_1, 1/prob_cens_t, sample_weights)

        return times.reshape(-1, 1), yc, sample_weights


class BaseYASGBT(BaseEstimator, SurvivalMixin, ABC):

    def __init__(
        self,
        sampling_strategy="uniform",
        n_iter=10,
        n_repetitions_per_iter=5,
        learning_rate=0.1,
        max_depth=7,
        min_samples_leaf=50,
        verbose=False,
        show_progressbar=True,
        random_state=None,
    ):
        self.sampling_strategy = sampling_strategy
        self.n_iter = n_iter
        self.n_repetitions_per_iter = n_repetitions_per_iter # TODO? data augmenting for early iterations
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        self.show_progressbar = show_progressbar
        self.random_state = random_state
    
    def fit(self, X, y, times=None, validation_data=None):

        # TODO: add check_X_y from sksurv
        monotonic_cst = np.zeros(X.shape[1]+1)
        monotonic_cst[0] = -1

        self.hgbt_ = self._get_model(monotonic_cst)

        data_sampler = IBSTrainingSampler(
            y,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )
        iterator = range(self.n_iter)
        if self.show_progressbar:
            iterator = tqdm(iterator)

        for idx_iter in iterator:
            times, yc, sample_weight = data_sampler.make_sample()
            Xt = np.hstack([times, X])
            self.hgbt_.max_iter += 1
            self.hgbt_.fit(Xt, yc, sample_weight=sample_weight)

            if self.verbose:
                train_ibs = self.compute_ibs(y, X_val=X)
                msg_ibs = f"round {idx_iter+1:03d} -- train ibs: {train_ibs:.6f}"
                
                if validation_data is not None:
                    X_val, y_val = validation_data
                    val_ibs = self.compute_ibs(y, X_val, y_val)
                    msg_ibs += f" -- val ibs: {val_ibs:.6f}"
                
                print(msg_ibs)
        
        return self
    
    def compute_ibs(self, y_train, X_val, y_val=None):
        if y_val is None:
            y_val = y_train
        times_val = get_time_grid(y_train, y_val)
        survival_probs = self.predict_survival_function(X_val, times_val)

        return integrated_brier_score(y_train, y_val, survival_probs, times_val)


class YASGBTClassifier(BaseYASGBT):

    name = "YASGBTClassifier"

    def _get_model(self, monotonic_cst):

        return HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=1,
            warm_start=True,
            monotonic_cst=monotonic_cst,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
        )

    def predict_survival_function(self, X, times):

        all_y_probs = []

        iterator = times
        if self.show_progressbar:
            iterator = tqdm(iterator)

        for t in iterator:
            t = np.full((X.shape[0], 1), t)
            Xt = np.hstack([t, X])
            y_probs = self.hgbt_.predict_proba(Xt)[:, 1]
            all_y_probs.append(y_probs)
        
        surv_probs = np.vstack(all_y_probs).T

        return surv_probs


class YASGBTRegressor(BaseYASGBT):

    name = "YASGBTRegressor"

    def _get_model(self, monotonic_cst):

        return HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=1,
                warm_start=True,
                monotonic_cst=monotonic_cst,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )

    def predict_survival_function(self, X, times):

        all_y_probs = []

        iterator = times
        if self.show_progressbar:
            iterator = tqdm(iterator)

        for t in iterator:
            t = np.full((X.shape[0], 1), t)
            Xt = np.hstack([t, X])
            y_probs = self.hgbt_.predict(Xt)
            all_y_probs.append(y_probs)
        
        surv_probs = np.vstack(all_y_probs).T

        return np.clip(surv_probs, 0, 1)
        
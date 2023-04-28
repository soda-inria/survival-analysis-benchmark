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
    
    def __init__(self, y, event_of_interest, sampling_strategy, random_state):
        self.y = y
        self.event_of_interest = event_of_interest
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        any_event = (y["event"] > 0).astype(bool)
        times, km = kaplan_meier_estimator(any_event, y["duration"])
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
        
        k = self.event_of_interest
        mask_y_k = (y["event"] == k) & (y["duration"] <= times)
        mask_y_0 = (y["event"] > 0) & (y["duration"] <= times) 
        mask_y_1 = y["duration"] > times 

        yc = np.zeros(y.shape[0])
        yc[mask_y_k] = 1

        y_any_event = np.empty(
            y.shape[0],
            dtype=[("event", bool), ("duration", float)],
        )
        y_any_event["event"] = (y["event"] > 0)
        y_any_event["duration"] = y["duration"]
        cens = CensoringDistributionEstimator().fit(y_any_event) 

        # calculate inverse probability of censoring weight at current time point t.
        prob_cens_t = cens.predict_proba(times)
        prob_cens_t = np.clip(prob_cens_t, 1e-8, 1)

        # calculate inverse probability of censoring weights at observed time point
        prob_cens_y = cens.predict_proba(y["duration"])
        prob_cens_y = np.clip(prob_cens_y, 1e-8, 1)

        sample_weights = np.where(mask_y_0, 1/prob_cens_y, 0)
        sample_weights = np.where(mask_y_1, 1/prob_cens_t, sample_weights)

        return times.reshape(-1, 1), yc, sample_weights


class GradientBoostedCIF(BaseEstimator, SurvivalMixin):
    """Histogram Gradient Boosted Tree to estimate the
    cause-specific Cumulative Incidence Function (CIF).

    Estimate a cause-specific CIF by computing the
    Brier Score for the kth cause of failure from _[1].
    One can obtain the survival probabilities any event
    by summing all cause-specific cif and computing 1 - sum.

    Parameters
    ----------
    event_of_interest : int, default=1
        The event to compute the CIF for. 0 always represent
        the censoring and must not be set.
    
    objective : {'ibs', 'inll'}, default='ibs'
        The objective of the model. In practise, both objective yields
        comparable results.
        
        - 'ibs' : integrated brier score. Use a `HistGradientBoostedRegressor`
          with the 'squared_error' loss. As we have no guarantee that the regression
          yields a survival function belonging to [0, 1], we clip the probabilities
          to this range.
        - 'inll' : integrated negative log likelihood. Use a
          `HistGradientBoostedClassifier` with 'log_loss' loss.
    
    TODO: complete the docstring.
          
    References
    ----------
    
    [1] M. Kretowska, "Tree-based models for survival data with competing risks",
        Computer Methods and Programs in Biomedicine 159 (2018) 185-198.
    """

    name = "GradientBoostedCIF"

    def __init__(
        self,
        event_of_interest=1,
        objective="ibs",
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
        self.event_of_interest = event_of_interest
        self.objective = objective
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
        monotonic_cst[0] = 1

        self.hgbt_ = self._get_model(monotonic_cst)

        data_sampler = IBSTrainingSampler(
            y,
            event_of_interest=self.event_of_interest,
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
    
    def predict_cumulative_incidence(self, X, times):
        all_y_cif = []

        iterator = times
        if self.show_progressbar:
            iterator = tqdm(iterator)

        for t in iterator:
            t = np.full((X.shape[0], 1), t)
            Xt = np.hstack([t, X])
            if self.objective == "ibs":
                y_cif = self.hgbt_.predict(Xt)
            else:
                y_cif = self.hgbt_.predict_proba(Xt)[:, 1] 
            all_y_cif.append(y_cif)
        
        cif = np.vstack(all_y_cif).T

        if self.objective == "ibs":
            cif = np.clip(cif, 0, 1)

        return cif
    
    def predict_survival_function(self, X, times):
        """Compute the event specific survival function.
        
        Warning: this metric only makes sense for binary or
        single event! To compute the survival function 
        for competitive events, one need to sum every cif from all 
        GradientBoostedCIF estimators.
        """
        return 1 - self.predict_cumulative_incidence(X, times)
    
    def _get_model(self, monotonic_cst):

        if self.objective == "ibs":
            return HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=1,
                warm_start=True,
                monotonic_cst=monotonic_cst,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
        elif self.objective == "inll":
            return HistGradientBoostingClassifier(
                loss="log_loss",
                max_iter=1,
                warm_start=True,
                monotonic_cst=monotonic_cst,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
        else:
            raise ValueError(
                "Parameter 'objective' must be either 'ibs' or 'inll', "
                f"got {self.objective}."
            )

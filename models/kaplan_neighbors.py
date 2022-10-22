import numpy as np

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from sksurv.tree.tree import _array_to_step_function

from xgbse.non_parametric import calculate_kaplan_vectorized

from .survival_mixin import SurvivalMixin


class KaplanNeighbors(BaseEstimator, SurvivalMixin):

    def __init__(self, neighbors_params=None):
        self.neighbors_params = neighbors_params
    
    def fit(self, X, y=None, time_bins=None):
        self.nearest_neighbors_ = NearestNeighbors(**self.neighbors_params).fit(X)
        self.y_train_ = y
        self.time_bins_ = time_bins
        return self
    
    def predict_survival_function(self, X, return_array=False):
        check_is_fitted(self, "nearest_neighbors_")
        X_idx = self.nearest_neighbors_.kneighbors(X, return_distance=False)
        y_preds = self.y_train_[X_idx]
        survival_probs, _, _ = calculate_kaplan_vectorized(
            E=y_preds["event"],
            T=y_preds["duration"],
            time_bins=self.time_bins_
        )
        survival_probs = survival_probs.values
        if return_array:
            return survival_probs
        return _array_to_step_function(self.time_bins_, survival_probs)

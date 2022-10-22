import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from sksurv.tree.tree import SurvivalTree, _array_to_step_function
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.functions import StepFunction

from .survival_mixin import SurvivalMixin


def _map_leaves_to_km(leaves, y_train, time_bins):
    unique_leaves = np.unique(leaves)
    leaf_to_km = dict()

    for leaf in unique_leaves:
        
        mask_leaf = np.where(leaves == leaf)
        _y_train = y_train[mask_leaf]
        times, km = kaplan_meier_estimator(_y_train["event"], _y_train["duration"])

        # build the grid by using the step function
        min_time, max_time = times[0], times[-1]
        mask_min_time = time_bins < min_time
        mask_max_time = time_bins > max_time
        mask_time = ~(mask_min_time | mask_max_time)
        km = StepFunction(times, km)(time_bins[mask_time])

        # fill time values
        min_km, max_km = km[0], km[-1]
        n_min_val = sum(mask_min_time)
        n_max_val = sum(mask_max_time)
        left_km = np.full(shape=n_min_val, fill_value=min_km)
        right_km = np.full(shape=n_max_val, fill_value=max_km)
        full_km = np.hstack([left_km, km, right_km])
        leaf_to_km[leaf] = full_km

    return leaf_to_km


class KaplanTree(BaseEstimator, SurvivalMixin):

    def fit(self, X, y=None, time_bins=None):
        leaves = self._get_leaves(X)
        self.leaf_to_km_ = _map_leaves_to_km(leaves, y, time_bins)
        self.time_bins_ = time_bins
        return self
    
    def predict_survival_function(self, X, return_array=False):
        check_is_fitted(self, "leaf_to_km_")
        x_leaves = self._get_leaves(X)
        survival_probs = np.vstack([self.leaf_to_km_[leaf] for leaf in x_leaves])
        if return_array:
            return survival_probs
        return _array_to_step_function(self.time_bins_, survival_probs)
        
    def _get_leaves(self, X):
        target_type = type_of_target(X)
        if target_type == "multilabel-indicator":
            X_leaves = np.argmax(X, axis=1)
        elif target_type == "multiclass":
            X_leaves = X
        else:
            raise ValueError(f"X must be a categorical label, got: {target_type}")
        x_leaves = self._ensure_leaves_1d(X_leaves)
        return x_leaves

    def _ensure_leaves_1d(self, X_leaves):
        if X_leaves.ndim == 2:
            x_leaves = np.asarray(X_leaves).ravel()  # scipy sparse yields a matrix, we need a ndarray
        else:
            x_leaves = X_leaves
        return x_leaves 
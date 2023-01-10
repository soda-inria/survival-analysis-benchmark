import numpy as np
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from sksurv.tree.tree import _array_to_step_function

from xgbse._debiased_bce import _build_multi_task_targets
from xgbse._base import DummyLogisticRegression

from .survival_mixin import SurvivalMixin


class MetaGridBC(BaseEstimator, SurvivalMixin):
    
    def __init__(self, classifier, n_jobs=None, verbose=0): 
        self.classifier = classifier
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X_train, y_train=None, time_bins=None):
        e_name, t_name = y_train.dtype.names
        targets_train, _ = _build_multi_task_targets(
            E=y_train[e_name],
            T=y_train[t_name],
            time_bins=time_bins,
        )
        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            estimators = parallel(
                delayed(self._fit_one_lr)(X_train, targets_train[:, i])
                for i in range(targets_train.shape[1])
            )
        self.time_bins_ = time_bins
        self.estimators_ = estimators
        return self

    def _fit_one_lr(self, X, target):
        mask = target != -1

        if len(target[mask]) == 0:
            # If there's no observation in a time bucket we raise an error
            raise ValueError("Error: No observations in a time bucket")
        elif len(np.unique(target[mask])) == 1:
            # If there's only one class in a time bucket
            # we create a dummy classifier that predicts that class and send a warning
            classifier = DummyLogisticRegression()
        else:
            classifier = clone(self.classifier)
        classifier.fit(X[mask, :], target[mask])
        return classifier
    
    def predict_survival_function(self, X, return_array=False):
        check_is_fitted(self, "estimators_")
        with Parallel(n_jobs=self.n_jobs) as parallel:
            y_preds = parallel(
                delayed(self._predict_one_lr)(X, estimator)
                for estimator in self.estimators_
            )
        y_preds = np.asarray(y_preds)
        survival_probs = np.cumprod(1 - y_preds, axis=0).T
        if return_array:
            return survival_probs
        return _array_to_step_function(self.time_bins_, survival_probs)
    
    def _predict_one_lr(self, X, estimator):
        y_pred = estimator.predict_proba(X)
        # return probability of "positive" event
        return y_pred[:, 1]
    
import warnings
import numpy as np
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import HistGradientBoostingClassifier

from sksurv.tree.tree import _array_to_step_function

from xgbse._debiased_bce import _build_multi_task_targets
from xgbse._base import DummyLogisticRegression

from .survival_mixin import SurvivalMixin


def make_bins(n_estimators, times):
    """Select `n_estimators` points uniformely from the vector `times`."""
    start, end = times[0], times[-1]
    bin_right_edges, period = np.linspace(start, end, n_estimators + 1, retstep=True)
    bin_centers = (bin_right_edges - period / 2)[1:].astype(int)
    return bin_centers, period


class IntegratedMetaGridBC(BaseEstimator, SurvivalMixin):
    def __init__(
        self,
        classifier=None,
        n_estimators=10,
        period=None,
        n_jobs=None,
        verbose=0,
        name="IntegratedMetaGridBC",
    ):
        self.classifier = classifier or HistGradientBoostingClassifier()
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.period = period
        self.verbose = verbose
        self.name = name

    def fit(self, X, y=None, times=None):

        if times[-1] != max(times):
            raise ValueError("`times` must be sorted")

        # compute n_estimators based on period
        n_estimators = None
        if self.period is not None:
            n_estimators = times[-1] // self.period + int(
                bool(len(times) % self.period)
            )
            if self.n_estimators is not None:
                warnings.warn(
                    f"`n_estimators`: {self.n_estimators} is superseded by `period`: {self.period}. "
                    f"`n_estimators` is now: {n_estimators}."
                )
        n_estimators = n_estimators or self.n_estimators

        if n_estimators > len(times):
            raise ValueError(
                "`n_estimators` is higher than the size "
                f"of `times` ({n_estimators} > {len(times)})"
            )

        bin_centers, period = make_bins(n_estimators, times)

        e_name, t_name = y.dtype.names
        targets_train, _ = _build_multi_task_targets(
            E=y[e_name],
            T=y[t_name],
            time_bins=bin_centers,
        )
        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            estimators = parallel(
                delayed(self._fit_one_lr)(X, targets_train[:, i])
                for i in range(len(bin_centers))
            )
        self.times_ = times
        self.bin_centers_ = bin_centers
        self.estimators_ = estimators
        self.n_estimators_ = n_estimators
        self.period_ = self.period or period
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

    def predict_survival_function(self, X, times=None, return_array=True):
        check_is_fitted(self, "estimators_")
        with Parallel(n_jobs=self.n_jobs) as parallel:
            y_preds_T = parallel(
                delayed(self._predict_one_lr)(X, estimator)
                for estimator in self.estimators_
            )
        # Probability of event = 1 in each time bins, conditional on no prior event.
        y_preds = np.asarray(y_preds_T).T  # (n_samples, n_estimators)
        survival_probs = np.cumprod(1 - y_preds, axis=1)  # (n_samples, n_estimators)
        hazards = y_preds / survival_probs
        integrated_survival_probs = self.get_integrated_survival_function(hazards)
        if return_array:
            return integrated_survival_probs
        return _array_to_step_function(self.times_, integrated_survival_probs)

    def _predict_one_lr(self, X, estimator):
        """Return probability of "positive" event.
        """
        y_pred = estimator.predict_proba(X)
        return y_pred[:, 1]

    def get_integrated_survival_function(self, hazards):
        """Compute the survival function from instantaneous hazards.
        """
        integrated_survival_probs = []
        for t in self.times_:
            idx_bin = int(t // self.period_)
            idx_bin = min(idx_bin, self.n_estimators_-1)
            t_trunc = idx_bin * self.period_
            cumulative_hazard = hazards[:, :idx_bin].sum(axis=1) * 1 + hazards[
                :, idx_bin
            ] * (t - t_trunc) / self.period_

            survival_proba = np.exp(-cumulative_hazard)
            integrated_survival_probs.append(survival_proba)
        return np.array(integrated_survival_probs).T
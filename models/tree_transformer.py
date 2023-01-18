import numpy as np
from numpy.testing import assert_array_equal
from pandas.core.frame import DataFrame

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state


# inspired from sklearn _gradient_boosting.pyx
def _random_sample_mask(
    n_sample,
    n_in_bag,
    random_state,
):
    sample_mask = np.hstack([
        np.zeros(n_sample - n_in_bag, dtype=bool),
        np.ones(n_in_bag, dtype=bool),
    ])
    random_state.shuffle(sample_mask)
    return sample_mask


class TreeTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        base_estimator=None,
        handle_survival_target=False,
        subsample=1.0,
        random_state=None,
    ):
        self.base_estimator = base_estimator
        self.handle_survival_target = handle_survival_target
        self.subsample = subsample
        self.random_state = random_state

    def fit_transform(self, X, y=None):

        if isinstance(X, DataFrame):
            raise TypeError("DataFrame not supported, convert it to a numpy ndarray")

        self._rng = check_random_state(self.random_state)

        if self.subsample < 1.0:
            n_sample = X.shape[0]
            n_in_bag = max(1, int(self.subsample * n_sample))
            sample_mask = _random_sample_mask(n_sample, n_in_bag, self._rng)
        else:
            sample_mask = np.ones(X.shape[0], dtype=bool)

        _y = y[sample_mask]

        y_event = _y[_y.dtype.names[0]]
        y_duration = _y[_y.dtype.names[1]]
        assert_array_equal(np.unique(y_event), [0, 1])
        y_duration = check_array(y_duration, dtype="numeric", ensure_2d=False)
        
        if self.base_estimator is None:
            if self.handle_survival_target:
                raise TypeError(
                    "Specify a base estimator that accepts a survival "
                    f"structure for y: {y.dtype}"
                )
            else:
                base_estimator = RandomForestRegressor()
        else:
            base_estimator = clone(self.base_estimator)
        
        if self.handle_survival_target:
            X_leaves = base_estimator.fit(X[sample_mask, :], _y).apply(X)
        else:
            # The base estimator will be fitted with a censoring bias
            X_leaves = base_estimator.fit(X[sample_mask, :], y_duration).apply(X)

        X_leaves = self._ensure_leaves_2d(X_leaves)
        self.encoder_ = OneHotEncoder(sparse=True, handle_unknown="ignore")
        X_ohe = self.encoder_.fit_transform(X_leaves)
        self.base_estimator_ = base_estimator
        return X_ohe

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, "base_estimator_")
        X_leaves = self.base_estimator_.apply(X)
        X_leaves = self._ensure_leaves_2d(X_leaves)
        return self.encoder_.transform(X_leaves)

    def _ensure_leaves_2d(self, X_leaves):
        if X_leaves.ndim == 1:
            X_leaves = X_leaves.reshape(-1, 1)
        return X_leaves
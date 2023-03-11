from abc import ABC, abstractmethod
import numpy as np

from models.survival_mixin import SurvivalMixin


class BaseWrapper(ABC, SurvivalMixin):

    def __init__(self, estimator, name=None, fit_kwargs=None):
        self.estimator = estimator
        self.name = name or estimator.__class__.__name__
        self.fit_kwargs = fit_kwargs or dict()
    
    @abstractmethod
    def fit(self, X_train, y_train, times=None):
        pass

    @abstractmethod
    def predict_survival_function(self, X_test, times):
        pass


class PipelineWrapper(BaseWrapper):
    
    def fit(self, X_train, y_train, times):
        last_est_name = self.estimator.steps[-1][0]
        times_kwargs = {f"{last_est_name}__times": times}
        self.estimator.fit(X_train, y_train, **times_kwargs)
    
    def predict_survival_function(self, X_test, times=None):
        return self.estimator.predict_survival_function(X_test)


class SkurvWrapper(BaseWrapper):

    def fit(self, X_train, y_train, times=None):
        self.estimator.fit(X_train, y_train)

    def predict_survival_function(self, X_test, times):
        step_funcs = self.estimator.predict_survival_function(X_test, return_array=False)
        survival_probs = np.vstack([step_func(times) for step_func in step_funcs])
        self.survival_probs_ = survival_probs
        return survival_probs


class XGBSEWrapper(BaseWrapper):
    
    def fit(self, X_train, y_train, times=None):
        self.estimator.fit(X_train, y_train, time_bins=times, **self.fit_kwargs)

    def predict_survival_function(self, X_test, times=None):
        survival_probs = self.estimator.predict(X_test, return_interval_probs=False)
        self.survival_probs_ = survival_probs
        return survival_probs


class DeepHitWrapper(BaseWrapper):

    def fit(self, X_train, y_train, times=None):
        self.estimator.fit(X_train, y_train, **self.fit_kwargs)

    def predict_survival_function(self, X_test, times=None):
        # StandardScaler
        X_test_trans = self.estimator[0].transform(X_test)

        # DeepHitestimator
        survival_probs = self.estimator[1].predict_surv_df(X_test_trans)
        time_index = np.asarray(survival_probs.index)
        survival_probs = survival_probs.reset_index(drop=True).T
        survival_probs.columns = time_index

        self.survival_probs_ = survival_probs
        return survival_probs
    
    def train_test_split(self, X, y, train_idxs, val_idxs):
        X_train = X[train_idxs, :]
        X_val = X[val_idxs, :]
        y_train = (
            y[0][train_idxs],
            y[1][train_idxs],
        )
        y_val = (
            y[0][val_idxs],
            y[1][val_idxs],
        )
        return X_train, y_train, X_val, y_val

    def prepare_y_scoring(self, y_train, y_val):
        """Convert y from PyCox format to scikit-survival.

        We test all estimators with scikit-survival score functions for exactness.
        """
        from sksurv.util import Surv

        y_train_arr = Surv.from_arrays(
            time=y_train[0],
            event=y_train[1],
            name_time="duration",
            name_event="event",
        )
        y_val_arr = Surv.from_arrays(
            time=y_val[0],
            event=y_val[1],
            name_time="duration",
            name_event="event",
        )
        return y_train_arr, y_val_arr

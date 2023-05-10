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
    
    def fit(self, X_train, y_train, times=None):
        last_est_name = self.estimator.steps[-1][0]
        times_kwargs = {f"{last_est_name}__times": times}
        self.estimator.fit(X_train, y_train, **times_kwargs)

    def predict_survival_function(self, X_test, times=None):
        return self.estimator.predict_survival_function(X_test, times=times)

    def predict_cumulative_incidence(self, X_test, times=None):
        transformers = self.estimator[:-1]
        X_test = transformers.transform(X_test)
        estimator = self.estimator[-1]
        return estimator.predict_cumulative_incidence(X_test, times=times)

    def predict_quantile(self, X_test, quantile=0.5, times=None):
        transformers = self.estimator[:-1]
        X_test = transformers.transform(X_test)
        estimator = self.estimator[-1]
        return estimator.predict_quantile(X_test, quantile=quantile, times=times)

    def predict_proba(self, X_test, time_horizon=None):
        transformers = self.estimator[:-1]
        X_test = transformers.transform(X_test)
        estimator = self.estimator[-1]
        return estimator.predict_proba(X_test, time_horizon=time_horizon)


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
        y_train_ = self.adapt_y(y_train)
        self.estimator.fit(X_train, y_train_, **self.fit_kwargs)

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
    
    def adapt_y(self, y):
        if not isinstance(y, tuple):
            y = (y["duration"], y["event"])
        y = (
            np.ascontiguousarray(y[0], dtype=int),
            np.ascontiguousarray(y[1], dtype=np.float32),
        )
        return y

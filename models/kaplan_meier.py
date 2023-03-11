import numpy as np

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.functions import StepFunction

from models.survival_mixin import SurvivalMixin


class KaplanMeier(SurvivalMixin):
    
    name = "KaplanMeier"
    
    def fit(self, X, y, times=None):
        self.km_x_, self.km_y_ = kaplan_meier_estimator(y["event"], y["duration"])
        return self
    
    def predict_survival_function(self, X, times):
        surv_probs = StepFunction(self.km_x_, self.km_y_)(times)
        return np.vstack([surv_probs] * X.shape[0])

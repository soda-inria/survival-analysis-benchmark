import numpy as np
from sksurv.tree.tree import _array_to_step_function


class SurvivalMixin:

    def predict_cumulative_hazard_function(self, X_test, return_array=False):
        survival_probs = self.predict_survival_function(X_test, return_array=True)
        cumulative_hazards = -np.log(survival_probs)
        if return_array:
            return cumulative_hazards
        return _array_to_step_function(self.time_bins_, cumulative_hazards)

    def predict(self, X_test):
        cumulative_hazards = self.predict_cumulative_hazard_function(
            X_test, return_array=True
        )
        return cumulative_hazards.sum(axis=1)

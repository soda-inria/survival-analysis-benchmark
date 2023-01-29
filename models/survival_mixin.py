import numpy as np
from sksurv.tree.tree import _array_to_step_function


class SurvivalMixin:

    def predict_cumulative_hazard_function(self, X_test, times):
        survival_probs = self.predict_survival_function(X_test, times)
        cumulative_hazards = -np.log(survival_probs + 1e-8)
        self.cumulative_hazards_ = cumulative_hazards 
        return cumulative_hazards

    def predict_risk_estimate(self, X_test, times):
        cumulative_hazards = self.predict_cumulative_hazard_function(X_test, times)
        return cumulative_hazards.sum(axis=1)

    @staticmethod
    def train_test_split(X, y, train_idxs, val_idxs):
        X_train, y_train = X.values[train_idxs, :], y[train_idxs]
        X_val, y_val = X.values[val_idxs, :], y[val_idxs]
        return X_train, y_train, X_val, y_val

    @staticmethod
    def prepare_y_scoring(y_train, y_val):
        """No-op"""
        return y_train, y_val

    #def score()
    # learning_rate, early_stopping? (avec val-set, split auto), max_iter, max_leaf_nodes, min_samples_leaf
from collections import namedtuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


ModelStage = namedtuple('ModelStage', ['model', 'labels'])


class MultiStageClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, stages, max_train_size=None, default_label='Benign'):
        self.default_label = default_label
        self.stages = stages
        self.max_train_size = max_train_size


    def fit(self, X, y):
        self.classes_ = np.array(y.cat.categories)
        previous_labels = set()

        for stage in self.stages:
            y_stage = y[~y.isin(previous_labels)]
            previous_labels |= stage.labels

            if self.max_train_size and len(y_stage) > self.max_train_size:
                y_stage = y_stage.sample(self.max_train_size)

            y_train = y_stage.where(y_stage.isin(stage.labels), self.default_label)
            X_train = X.loc[y_train.index]

            stage.model.fit(X_train, y_train)

        return self


    def predict_proba(self, X):
        index_default = list(self.classes_).index(self.default_label)
        proba = np.zeros(shape=(len(X), len(self.classes_)))
        proba[:, index_default] = 1.0

        previous_labels = {}

        for stage in self.stages:
            stage_mask = proba.max(axis=1) == proba[:, index_default]
            stage_idx = stage_mask.nonzero()[0]

            idx = [
                list(self.classes_).index(label)
                for label in stage.model.classes_
            ]

            previous_default_proba = proba[stage_idx, index_default][:, None]
            stage_proba = stage.model.predict_proba(X[stage_mask])

            proba[stage_idx[:, None], idx] = previous_default_proba * stage_proba

        return proba


    def predict(self, X):
        proba = self.predict_proba(X)

        return self.classes_[proba.argmax(1)]

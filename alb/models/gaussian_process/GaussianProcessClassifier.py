#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from mgktools.models.classification.gpc.gpc import GaussianProcessClassifier


class GPClassifier(GaussianProcessClassifier):
    def fit(self, train_data, sample_weight=None):
        X = train_data.X
        y = train_data.y
        if y.ndim == 2:
            assert y.shape[1] == 1
            y = y.ravel()
        return super().fit(X, y)

    def predict_uncertainty(self, pred_data):
        X = pred_data.X
        preds = self.predict_proba(X).reshape(-1, 1)
        preds = np.concatenate([preds, 1 - preds], axis=1)
        return 0.25 - np.var(preds, axis=1)

    def predict_value(self, pred_data):
        X = pred_data.X
        return super().predict_proba(X)

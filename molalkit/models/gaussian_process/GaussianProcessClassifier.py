#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from mgktools.models.classification.gpc.gpc import GaussianProcessClassifier
from molalkit.models.base import BaseSklearnModel


class GPClassifier(GaussianProcessClassifier, BaseSklearnModel):
    def fit_alb(self, train_data):
        return self.fit_alb_(train_data, self)

    def predict_uncertainty(self, pred_data):
        return self.predict_uncertainty_c(pred_data, self)

    def predict_value(self, pred_data):
        X = pred_data.X
        return super().predict_proba(X)

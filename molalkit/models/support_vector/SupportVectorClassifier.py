#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from molalkit.models.base import BaseSklearnModel


class SVClassifier(SVC, BaseSklearnModel):
    def fit_alb(self, train_data):
        return self.fit_alb_(train_data, self)

    def predict_uncertainty(self, pred_data):
        return self.predict_uncertainty_c(pred_data, self)

    def predict_value(self, pred_data):
        return self.predict_value_c(pred_data, self)

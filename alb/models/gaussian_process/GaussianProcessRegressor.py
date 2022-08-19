#!/usr/bin/env python
# -*- coding: utf-8 -*-
from graphdot.model.gaussian_process import GaussianProcessRegressor as GPR


class GPRegressor(GPR):
    def fit(self, train_data, **kwargs):
        X = train_data.X
        y = train_data.y
        super().fit(X, y, **kwargs)

    def predict_uncertainty(self, pred_data):
        X = pred_data.X
        return self.predict(X, return_std=True)[1]

    def predict_value(self, pred_data):
        X = pred_data.X
        return self.predict(X)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def fit_alb(self, data):
        pass

    @abstractmethod
    def predict_value(self, data):
        pass

    @abstractmethod
    def predict_uncertainty(self, data):
        pass


class BaseSklearnModel(BaseModel, ABC):
    @staticmethod
    def fit_alb_(train_data, sklearn_model):
        X = train_data.X
        y = train_data.y
        if y.ndim == 2:
            assert y.shape[1] == 1
            y = y.ravel()
        return sklearn_model.fit(X, y)

    @staticmethod
    def predict_uncertainty_c(pred_data, model):
        X = pred_data.X
        p = model.predict_proba(X)
        return (0.25 - np.var(p, axis=1)) * 4

    @staticmethod
    def predict_value_c(pred_data, model):
        X = pred_data.X
        return model.predict_proba(X)[:, 1]

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from chemprop.models.mpnn import MPNN as MPNNBase


class MPNN(MPNNBase):
    def fit_alb(self, *args, **kwargs):
        super().fit(*args, **kwargs)

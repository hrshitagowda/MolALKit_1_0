#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import copy
import os
import pickle
import numpy as np
import pandas as pd
from random import Random
import rdkit.Chem.AllChem as Chem
from chemprop.data.scaffold import scaffold_to_smiles
from logging import Logger
from .data import Dataset


def get_data(path: str,
             pure_columns: List[str] = None,
             mixture_columns: List[str] = None,
             target_columns: List[str] = None,
             feature_columns: List[str] = None,
             features_generator: List[str] = None,
             features_combination: Literal['concat', 'mean'] = None,
             n_jobs: int = 8):
    df = pd.read_csv(path)
    return Dataset.from_dataframe(df,
                                  pure_columns=pure_columns,
                                  mixture_columns=mixture_columns,
                                  target_columns=target_columns,
                                  feature_columns=feature_columns,
                                  features_generator=features_generator,
                                  features_combination=features_combination,
                                  n_jobs=n_jobs)

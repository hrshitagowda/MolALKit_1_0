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


def get_split_sizes(n_samples: int,
                    split_ratio: List[float]):
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError(f"Split split_ratio do not sum to 1. Received splits: {split_ratio}")
    if any([size < 0 for size in split_ratio]):
        raise ValueError(f"Split split_ratio must be non-negative. Received splits: {split_ratio}")

    acc_ratio = np.cumsum([0.] + split_ratio)
    split_sizes = [math.ceil(acc_ratio[i + 1] * n_samples) - math.ceil(acc_ratio[i] * n_samples)
                   for i in range(len(acc_ratio) - 1)]
    assert sum(split_sizes) == n_samples
    return split_sizes


def split_data(n_samples: int,
               smiles: List[str] = None,
               targets: List = None,
               split_type: Literal['random', 'scaffold_order', 'scaffold_random', 'class'] = 'random',
               sizes: List[float] = (0.8, 0.2),
               n_samples_per_class: int = 1,
               seed: int = 0,
               logger: Logger = None):
    if logger is not None:
        info = logger.info
        warn = logger.warning
    else:
        info = print
        warn = print

    random = Random(seed)
    np.random.seed(seed)
    split_index = [[] for size in sizes]
    if split_type in ['scaffold_random', 'scaffold_order']:
        index_size = get_split_sizes(n_samples, split_ratio=sizes)
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        scaffold_to_indices = scaffold_to_smiles(mols, use_indices=True)
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

        scaffold_count = [0 for size in sizes]
        index = [0, 1]
        for index_set in index_sets:
            if split_type == 'scaffold_random':
                random.shuffle(index)
            for i in index:
                s_index = split_index[i]
                if len(s_index) + len(index_set) <= index_size[i]:
                    s_index += index_set
                    scaffold_count[i] += 1
                    break
            else:
                split_index[0] += index_set
        info(f'Total scaffolds = {len(scaffold_to_indices):,} | ')
        for i, count in enumerate(scaffold_count):
            info(f'split {i} scaffolds = {count:,} | ')
    elif split_type == 'random':
        indices = list(range(n_samples))
        random.shuffle(indices)
        index_size = get_split_sizes(n_samples, split_ratio=sizes)
        end = 0
        for i, size in enumerate(index_size):
            start = end
            end = start + size
            split_index[i] = indices[start:end]
    elif split_type == 'class':
        class_list = np.unique(targets)
        assert len(class_list) > 1
        num_class = len(class_list)
        if num_class > 10:
            warn('You are splitting a classification dataset with more than 10 classes.')
        if n_samples_per_class is None:
            assert len(sizes) == 2
            n_samples_per_class = int(sizes[0] * n_samples / num_class)
            assert n_samples_per_class > 0

        for c in class_list:
            index = []
            for i, t in enumerate(targets):
                if t == c:
                    index.append(i)
            split_index[0].extend(np.random.choice(index, n_samples_per_class, replace=False).tolist())
        for i in range(n_samples):
            if i not in split_index[0]:
                split_index[1].append(i)
    else:
        raise ValueError(f'split_type "{split_type}" not supported.')
    assert sum([len(i) for i in split_index]) == n_samples
    return split_index

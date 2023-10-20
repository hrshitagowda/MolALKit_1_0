#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple, Callable
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding


def get_topn_idx(values: np.ndarray, n: int = 1, target: Union[Literal['max', 'min'], float] = 'max',
                 cutoff: float = None) -> List[int]:
    if isinstance(values, list):
        values = np.array(values)
    if target == 'min':
        values = - values
        if cutoff is not None:
            cutoff = - cutoff
    elif isinstance(target, float):
        assert cutoff is None
        values = np.absolute(values - target)  # distance from target
    if cutoff is not None:
        n_candidates = len(np.where(values >= cutoff)[0])
        n = min(n, n_candidates)
        if n == 0:
            return []
    # Includes tiny random values to randomly sort duplicated values
    sorting_key = values + np.random.random(len(values)) * 1e-10
    return np.argsort(sorting_key)[-n:].tolist()


class BaseSelectionMethod(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def info(self) -> str:
        pass


class BaseRandomSelectionMethod(BaseSelectionMethod, ABC):
    """Base SelectionMethod that uses random seed."""
    def __init__(self, batch_size: int = 1, seed: int = 0):
        self.batch_size = batch_size
        np.random.seed(seed)


class BaseClusterSelectionMethod(BaseRandomSelectionMethod, ABC):
    def __init__(self, batch_size: int = 1, cluster_size: int = None, seed: int = 0):
        super().__init__(batch_size=batch_size, seed=seed)
        if cluster_size is None:
            self.cluster_size = batch_size * 20
        else:
            self.cluster_size = cluster_size

    @staticmethod
    def find_distant_samples(gram_matrix: List[List[float]], batch_size: int = 1) -> List[int]:
        """ Find distant samples from a pool using clustering method.

        Parameters
        ----------
        gram_matrix: gram (kernel) matrix of the samples.
        batch_size: number of samples to be selected.

        Returns
        -------
        List of idx
        """
        embedding = SpectralEmbedding(
            n_components=batch_size,
            affinity='precomputed'
        ).fit_transform(gram_matrix)

        cluster_result = KMeans(
            n_clusters=batch_size,
            # random_state=self.args.seed
        ).fit_predict(embedding)
        # find all center of clustering
        center = np.array([embedding[cluster_result == i].mean(axis=0)
                           for i in range(batch_size)])
        total_distance = defaultdict(dict)  # (key: cluster_idx, val: dict of (key:sum of distance, val:idx))
        for i in range(len(cluster_result)):
            cluster_class = cluster_result[i]
            total_distance[cluster_class][((np.square(
                embedding[i] - np.delete(center, cluster_class, axis=0))).sum(
                axis=1) ** -0.5).sum()] = i
        add_idx = [total_distance[i][min(total_distance[i].keys())] for i in
                   range(batch_size)]  # find min-in-cluster-distance associated idx
        return add_idx


class BaseIterativeSelectionMethod(BaseSelectionMethod, ABC):
    pass


class RandomSelectionMethod(BaseRandomSelectionMethod):
    def __call__(self, data_pool, **kwargs) -> Tuple[List[int], None]:
        if self.batch_size < len(data_pool):
            return np.random.choice(range(len(data_pool)), self.batch_size, replace=False).tolist(), None
        else:
            return list(range(len(data_pool))), None

    @property
    def info(self) -> str:
        return f'RandomSelectionMethod(batch_size={self.batch_size})'


class ClusterRandomSelectionMethod(BaseClusterSelectionMethod):
    def __call__(self, data_pool, kernel: Callable) -> Tuple[List[int], None]:
        assert self.batch_size < len(data_pool)
        idx_candidates = np.random.choice(range(len(data_pool)), self.cluster_size, replace=False).tolist()
        K = kernel(data_pool.X[idx_candidates])
        add_idx = self.find_distant_samples(gram_matrix=K, batch_size=self.batch_size)
        idx = np.array(idx_candidates)[add_idx]
        return idx, None

    @property
    def info(self) -> str:
        return f'ClusterRandomSelectionMethod(batch_size={self.batch_size}, cluster_size={self.cluster_size})'


class ExplorativeSelectionMethod(BaseRandomSelectionMethod):
    def __call__(self, model, data_pool, **kwargs) -> Tuple[List[int], List[float]]:
        y_std = model.predict_uncertainty(data_pool)
        idx = get_topn_idx(y_std, n=self.batch_size)
        acquisition = y_std[np.array(idx)].tolist()
        return idx, acquisition

    @property
    def info(self) -> str:
        return f'ExplorativeSelectionMethod(batch_size={self.batch_size})'


class ClusterExplorativeSelectionMethod(BaseClusterSelectionMethod):
    def __call__(self, model, data_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float]]:
        y_std = model.predict_uncertainty(data_pool)
        idx_candidates = get_topn_idx(y_std, n=self.cluster_size)
        K = kernel(data_pool.X[idx_candidates])
        add_idx = self.find_distant_samples(gram_matrix=K, batch_size=self.batch_size)
        idx = np.array(idx_candidates)[add_idx]
        acquisition = y_std[idx].tolist()
        return idx, acquisition

    @property
    def info(self) -> str:
        return f'ClusterExplorativeSelectionMethod(batch_size={self.batch_size}, cluster_size={self.cluster_size})'


class ExploitiveSelectionMethod(BaseRandomSelectionMethod):
    def __init__(self, target, seed: int = 0):
        super().__init__(seed=seed)
        self.target = target

    def __call__(self, model, data_pool, **kwargs) -> Tuple[List[int], List[float]]:
        y_pred = model.predict_value(data_pool)
        idx = get_topn_idx(y_pred, n=self.batch_size, target=self.target)
        acquisition = y_pred[np.array(idx)].tolist()
        return idx, acquisition

    @property
    def info(self) -> str:
        return f'ExploitiveSelectionMethod(batch_size={self.batch_size}, target={self.target})'


class ClusterExploitiveSelectionMethod(BaseClusterSelectionMethod):
    def __init__(self, target, seed: int = 0):
        super().__init__(seed=seed)
        self.target = target

    def __call__(self, model, data_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float]]:
        y_pred = model.predict_value(data_pool)
        idx_candidates = get_topn_idx(y_pred, n=self.cluster_size, target=self.target)
        K = kernel(data_pool.X[idx_candidates])
        add_idx = self.find_distant_samples(gram_matrix=K, batch_size=self.batch_size)
        idx = np.array(idx_candidates)[add_idx]
        acquisition = y_pred[idx].tolist()
        return idx, acquisition

    @property
    def info(self) -> str:
        return (f'ClusterExploitiveSelectionMethod(batch_size={self.batch_size}, '
                f'cluster_size={self.cluster_size}, target={self.target})')


class ProbabilityImprovementSelectionMethod(BaseRandomSelectionMethod):
    pass


class ExpectedImprovementSelectionMethod(BaseRandomSelectionMethod):
    pass


class UpperConfidenceBoundSelectionMethod(BaseRandomSelectionMethod):
    pass

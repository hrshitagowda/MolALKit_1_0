#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple, Callable
import copy
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding


def get_topn_idx(values: np.ndarray, n: int = 1, target: Union[Literal['max', 'min'], float] = 'max',
                 cutoff: float = None) -> List[int]:
    """ Get the indices of top n values.

    Parameters
    ----------
    values: array-like.
    n: number of indices to be selected.
    target: 'max', 'min', or a float value.
    cutoff: if not None, only values >= cutoff (when target=max) will be considered.

    Returns
    -------

    """
    if isinstance(values, list):
        values = np.array(values)
    if target == 'min':
        values = - values
        if cutoff is not None:
            cutoff = - cutoff
    elif isinstance(target, float):
        assert cutoff is None
        values = - np.absolute(values - target)  # distance from target
    if cutoff is not None:
        n_candidates = len(np.where(values >= cutoff)[0])
        n = min(n, n_candidates)
        if n == 0:
            return []
    # Includes tiny random values to randomly sort duplicated values
    sorting_key = values + np.random.random(len(values)) * 1e-10
    return np.argsort(sorting_key)[-n:].tolist()


def get_subset(dataset, idx):
    subset = copy.deepcopy(dataset)
    subset.data = [data for i, data in enumerate(dataset.data) if i in idx]
    return subset


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

    def get_idx_cluster(self, data_pool, kernel: Callable, idx_candidates) -> List[int]:
        """ Find distant samples from a pool using KMeans clustering method."""
        K = kernel(data_pool.X[idx_candidates])
        add_idx = self.find_distant_samples(gram_matrix=K, batch_size=self.batch_size)
        return add_idx

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


class BasePartialQuerySelectionMethod(BaseSelectionMethod, ABC):
    @staticmethod
    def get_partial_data(data_pool, n_query):
        # randomly select n_query samples from data_pool.
        if len(data_pool) <= n_query:
            return data_pool, np.array(range(len(data_pool)))
        else:
            query_idx = np.random.choice(range(len(data_pool)), n_query, replace=False)
            # get data_query
            data_query = get_subset(data_pool, query_idx)
            return data_query, query_idx


class BaseIterativeSelectionMethod(BaseSelectionMethod, ABC):
    pass


class RandomSelectionMethod(BaseRandomSelectionMethod):
    def __call__(self, data_pool, **kwargs) -> Tuple[List[int], None, List[int]]:
        if self.batch_size < len(data_pool):
            idx = np.random.choice(range(len(data_pool)), self.batch_size, replace=False).tolist()
            mask = np.isin(range(len(data_pool)), idx)
            idx_remain = np.arange(len(data_pool))[~mask].tolist()
            return idx, None, idx_remain
        else:
            return list(range(len(data_pool))), None, []

    @property
    def info(self) -> str:
        return f'RandomSelectionMethod(batch_size={self.batch_size})'


class ClusterRandomSelectionMethod(BaseClusterSelectionMethod):
    def __call__(self, data_pool, kernel: Callable, **kwargs) -> Tuple[List[int], None, List[int]]:
        if self.batch_size < len(data_pool):
            # get idx for samples to conduct clustering
            if self.cluster_size < len(data_pool):
                idx_candidates = np.random.choice(range(len(data_pool)), self.cluster_size, replace=False)
            else:
                idx_candidates = np.arange(len(data_pool))
            # get idx after clustering, each cluster has one sample
            idx_ = np.array(self.get_idx_cluster(data_pool, kernel, idx_candidates))
            idx = idx_candidates[idx_].tolist()
            # get idx for the remaining samples
            mask = np.isin(range(len(data_pool)), idx)
            idx_remain = np.arange(len(data_pool))[~mask].tolist()
            return idx, None, idx_remain
        else:
            return list(range(len(data_pool))), None, []

    @property
    def info(self) -> str:
        return f'ClusterRandomSelectionMethod(batch_size={self.batch_size}, cluster_size={self.cluster_size})'


class ExplorativeSelectionMethod(BaseRandomSelectionMethod):
    def __call__(self, model, data_pool, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        # get predicted uncertainty
        y_std = model.predict_uncertainty(data_pool)
        # get idx for selected samples
        idx = get_topn_idx(y_std, n=self.batch_size)
        # get predicted uncertainty for the selected samples
        acquisition = y_std[np.array(idx)].tolist()
        # get idx for the remaining samples
        mask = np.isin(range(len(data_pool)), idx)
        idx_remain = np.arange(len(data_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f'ExplorativeSelectionMethod(batch_size={self.batch_size})'


class ExplorativeParitialQuerySelectionMethod(BaseRandomSelectionMethod, BasePartialQuerySelectionMethod):
    def __init__(self, n_query: int, batch_size: int = 1, seed: int = 0):
        super().__init__(batch_size=batch_size, seed=seed)
        self.n_query = n_query
        assert batch_size <= n_query

    def __call__(self, model, data_pool, stop_cutoff, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        # get partial data for active learning. Other data won't be considered in this iteration.
        data_query, query_idx = self.get_partial_data(data_pool, self.n_query)
        mask = np.isin(range(len(data_pool)), query_idx)
        idx_remain = np.arange(len(data_pool))[~mask].tolist()
        # get predicted uncertainty
        y_std = model.predict_uncertainty(data_query)
        # get idx and predicted uncertainty for selected samples
        idx_ = np.array(get_topn_idx(y_std, n=self.batch_size))
        acquisition = y_std[idx_].tolist()
        idx = query_idx[idx_].tolist()
        if stop_cutoff is not None:
            # confident samples will be removed in the pool set and will not be considered in the future AL.
            idx_unconfident = query_idx[np.where(y_std > stop_cutoff)[0]]
            mask = np.isin(idx_unconfident, idx)
            idx_remain += idx_unconfident[~mask].tolist()
        else:
            mask = np.isin(range(len(data_pool)), idx)
            idx_remain = np.arange(len(data_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f'ExplorativeParitialQuerySelectionMethod(batch_size={self.batch_size}, n_query={self.n_query})'


class ClusterExplorativeSelectionMethod(BaseClusterSelectionMethod):
    def __call__(self, model, data_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        # get predicted uncertainty
        y_std = model.predict_uncertainty(data_pool)
        # get idx for samples to conduct clustering
        idx_candidates = get_topn_idx(y_std, n=self.cluster_size)
        # get idx after clustering, each cluster has one sample
        if len(idx_candidates) < self.batch_size:
            idx = idx_candidates
        else:
            idx_ = np.array(self.get_idx_cluster(data_pool, kernel, idx_candidates))
            idx = np.array(idx_candidates)[idx_].tolist()
        # get predicted uncertainty for the selected samples
        acquisition = y_std[np.array(idx)].tolist()
        # get idx for the remaining samples
        mask = np.isin(range(len(data_pool)), idx)
        idx_remain = np.arange(len(data_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f'ClusterExplorativeSelectionMethod(batch_size={self.batch_size}, cluster_size={self.cluster_size})'


class ClusterExplorativeParitialQuerySelectionMethod(BaseClusterSelectionMethod, BasePartialQuerySelectionMethod):
    def __init__(self, n_query: int, batch_size: int = 1, cluster_size: int = None, seed: int = 0):
        super().__init__(batch_size=batch_size, cluster_size=cluster_size, seed=seed)
        self.n_query = n_query

    def __call__(self, model, data_pool, kernel: Callable, stop_cutoff, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        # get partial data for active learning. Other data won't be considered in this iteration.
        data_query, query_idx = self.get_partial_data(data_pool, self.n_query)
        mask = np.isin(range(len(data_pool)), query_idx)
        idx_remain = np.arange(len(data_pool))[~mask].tolist()
        # get predicted uncertainty
        y_std = model.predict_uncertainty(data_query)
        # get idx for samples to conduct clustering
        idx_candidates_ = np.array(get_topn_idx(y_std, n=self.cluster_size))
        idx_candidates = query_idx[idx_candidates_]
        # get idx after clustering, each cluster has one sample
        if len(idx_candidates) < self.batch_size:
            idx_ = np.arange(len(idx_candidates))
        else:
            idx_ = np.array(self.get_idx_cluster(data_pool, kernel, idx_candidates))
        # get idx for selected and confident samples
        idx = idx_candidates[idx_].tolist()
        acquisition = y_std[idx_candidates_[idx_]].tolist()
        if stop_cutoff is not None:
            # confident will be removed in the pool set and will not be considered in the future.
            idx_unconfident = query_idx[np.where(y_std > stop_cutoff)[0]]
            mask = np.isin(idx_unconfident, idx)
            idx_remain += idx_unconfident[~mask].tolist()
        else:
            mask = np.isin(range(len(data_pool)), idx)
            idx_remain = np.arange(len(data_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return (f'ExplorativeParitialQueryClusterSelectionMethod(batch_size={self.batch_size}, '
                f'cluster_size={self.cluster_size}, n_query={self.n_query})')


class ExploitiveSelectionMethod(BaseRandomSelectionMethod):
    def __init__(self, target, batch_size: int = 1, seed: int = 0):
        super().__init__(batch_size=batch_size, seed=seed)
        self.target = target

    def __call__(self, model, data_pool, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        y_pred = model.predict_value(data_pool)
        idx = get_topn_idx(y_pred, n=self.batch_size, target=self.target)
        acquisition = y_pred[np.array(idx)].tolist()
        mask = np.isin(range(len(data_pool)), idx)
        idx_remain = np.arange(len(data_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

    @property
    def info(self) -> str:
        return f'ExploitiveSelectionMethod(batch_size={self.batch_size}, target={self.target})'


class ClusterExploitiveSelectionMethod(BaseClusterSelectionMethod):
    def __init__(self, target, batch_size: int = 1, cluster_size: int = None, seed: int = 0):
        super().__init__(batch_size=batch_size, cluster_size=cluster_size, seed=seed)
        self.target = target

    def __call__(self, model, data_pool, kernel: Callable, **kwargs) -> Tuple[List[int], List[float], List[int]]:
        # get predicted value
        y_pred = model.predict_value(data_pool)
        # get idx for samples to conduct clustering
        idx_candidates = get_topn_idx(y_pred, n=self.cluster_size, target=self.target)
        # get idx after clustering, each cluster has one sample
        if len(idx_candidates) < self.batch_size:
            idx = idx_candidates
        else:
            idx_ = np.array(self.get_idx_cluster(data_pool, kernel, idx_candidates))
            idx = np.array(idx_candidates)[idx_].tolist()
        # get predicted value for the selected samples
        acquisition = y_pred[np.array(idx)].tolist()
        # get idx for the remaining samples
        mask = np.isin(range(len(data_pool)), idx)
        idx_remain = np.arange(len(data_pool))[~mask].tolist()
        return idx, acquisition, idx_remain

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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import json
import pickle
import pandas as pd
import numpy as np
from logging import Logger
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
from ..args import Metric
from tap import Tap


def eval_metric_func(y, y_pred, metric: str) -> float:
    if metric == 'roc-auc':
        return roc_auc_score(y, y_pred)
    elif metric == 'accuracy':
        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        return accuracy_score(y, y_pred)
    elif metric == 'precision':
        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        return precision_score(y, y_pred, average='macro')
    elif metric == 'recall':
        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        return recall_score(y, y_pred, average='macro')
    elif metric == 'f1_score':
        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        return f1_score(y, y_pred, average='macro')
    elif metric == 'mcc':
        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        return matthews_corrcoef(y, y_pred)
    elif metric == 'r2':
        return r2_score(y, y_pred)
    elif metric == 'mae':
        return mean_absolute_error(y, y_pred)
    elif metric == 'mse':
        return mean_squared_error(y, y_pred)
    elif metric == 'rmse':
        return np.sqrt(eval_metric_func(y, y_pred, 'mse'))
    elif metric == 'max':
        return np.max(abs(y - y_pred))
    else:
        raise RuntimeError(f'Unsupported metrics {metric}')


class ActiveLearner:
    def __init__(self,
                 save_dir: str,
                 dataset_type: Literal['regression', 'classification', 'multiclass'],
                 metrics: List[Metric],
                 learning_type: Literal['explorative', 'exploitive', 'EI', 'passive'],
                 model_selector,
                 dataset_train_selector,
                 dataset_pool_selector,
                 dataset_val_evaluator,
                 batch_size: int = 1,
                 batch_algorithm: Literal['nlargest', 'cluster'] = 'nlargest',
                 stop_size=None,
                 kernel=None,
                 cluster_size=None,
                 model_evaluator=None,
                 dataset_train_evaluator=None,
                 dataset_pool_evaluator=None,
                 model_extra_evaluators=None,
                 dataset_train_extra_evaluators=None,
                 dataset_pool_extra_evaluators=None,
                 dataset_val_extra_evaluators=None,
                 evaluate_stride: int = None,
                 extra_evaluators_only: bool = False,
                 save_cpt_stride: int = None,
                 seed: int = 0,
                 logger: Logger = None):
        self.save_dir = save_dir
        self.dataset_type = dataset_type
        self.metrics = metrics
        self.learning_type = learning_type
        self.batch_size = batch_size
        self.batch_algorithm = batch_algorithm
        self.stop_size = stop_size
        self.kernel = kernel
        self.cluster_size = cluster_size
        self.model_selector = model_selector
        self.model_evaluator_ = model_evaluator
        self.dataset_train_selector = dataset_train_selector
        self.dataset_pool_selector = dataset_pool_selector
        self.dataset_train_evaluator_ = dataset_train_evaluator
        self.dataset_pool_evaluator_ = dataset_pool_evaluator
        self.dataset_val_evaluator = dataset_val_evaluator
        self.model_extra_evaluators = model_extra_evaluators or []
        self.dataset_train_extra_evaluators = dataset_train_extra_evaluators or []
        self.dataset_pool_extra_evaluators = dataset_pool_extra_evaluators or []
        self.dataset_val_extra_evaluators = dataset_val_extra_evaluators or []

        self.evaluate_stride = evaluate_stride
        self.extra_evaluators_only = extra_evaluators_only
        self.save_cpt_stride = save_cpt_stride

        self.seed = seed
        if logger is not None:
            self.info = logger.info
        else:
            self.info = print
        self.n_iter = 0

    @property
    def active_learning_traj_dict(self) -> Dict:
        if not hasattr(self, '_active_learning_traj_dict'):
            self._active_learning_traj_dict = {'training_size': [], 'acquisition': []}
            for metric in self.metrics:
                self._active_learning_traj_dict[metric] = []
        return self._active_learning_traj_dict

    @property
    def active_learning_traj_extra_dict(self) -> List[Dict]:
        if not hasattr(self, '_active_learning_traj_extra_dict'):
            self._active_learning_traj_extra_dict = \
                [{'training_size': []} for i in range(len(self.model_extra_evaluators))]
            for metric in self.metrics:
                for alt in self._active_learning_traj_extra_dict:
                    alt[metric] = []
        return self._active_learning_traj_extra_dict

    @property
    def model_evaluator(self):
        if self.model_evaluator_ is None:
            return self.model_selector
        else:
            return self.model_evaluator_

    @property
    def dataset_train_evaluator(self):
        if self.dataset_train_evaluator_ is None:
            return self.dataset_train_selector
        else:
            return self.dataset_train_evaluator_

    @property
    def dataset_pool_evaluator(self):
        if self.dataset_pool_evaluator_ is None:
            return self.dataset_pool_selector
        else:
            return self.dataset_pool_evaluator_

    @property
    def yoked_learning(self) -> bool:
        """use different models for data selection and performance evaluation"""
        if self.model_evaluator_ is not None:
            return True
        else:
            return False

    @property
    def train_size(self) -> int:
        return len(self.dataset_train_selector)

    @property
    def pool_size(self) -> int:
        return len(self.dataset_pool_selector)

    def termination(self) -> bool:
        if self.pool_size == 0:
            return True
        elif self.stop_size is not None and self.train_size >= self.stop_size:
            return True
        else:
            return False

    def run(self):
        self.info('start active learning with training set size = %d' % self.train_size)
        self.info('pool set size = %d' % self.pool_size)
        for n_iter in range(self.n_iter, self.n_iter + self.pool_size):
            if self.termination():
                break
            self.info('Start an new iteration of active learning: %d.' % n_iter)
            # training
            self.model_selector.fit(self.dataset_train_selector)
            # add sample
            self.add_samples()
            # evaluate
            if self.evaluate_stride is not None and self.train_size % self.evaluate_stride == 0:
                self.evaluate()
            if self.save_cpt_stride is not None and n_iter % self.save_cpt_stride == 0:
                self.n_iter = n_iter + 1
                self.save(path=self.save_dir, overwrite=True)
                self.info('save checkpoint file %s/al.pkl' % self.save_dir)
            self.info('Training set size = %i' % self.train_size)
            self.info('Pool set size = %i' % self.pool_size)
        if self.active_learning_traj_dict['training_size'][-1] != self.train_size:
            self.evaluate()

    def evaluate(self):
        self.info('evaluating model performance.')
        if not self.extra_evaluators_only:
            if self.yoked_learning:
                self.model_evaluator.fit(self.dataset_train_evaluator)
            y_pred = self.model_evaluator.predict_value(self.dataset_val_evaluator)

            self.active_learning_traj_dict['training_size'].append(self.train_size)
            if hasattr(self, 'acquisition'):
                self.active_learning_traj_dict['acquisition'].append(json.dumps(self.acquisition))
            else:
                self.active_learning_traj_dict['acquisition'].append('none')
            for metric in self.metrics:
                metric_value = eval_metric_func(self.dataset_val_evaluator.y, y_pred, metric=metric)
                self.info('Evaluation performance %s: %.5f' % (metric, metric_value))
                self.active_learning_traj_dict[metric].append(metric_value)
            pd.DataFrame(self.active_learning_traj_dict).to_csv('%s/active_learning.traj' % self.save_dir, index=False)
        self.evaluate_extra()
        self.info('evaluating model performance finished.')

    def evaluate_extra(self):
        for i, model in enumerate(self.model_extra_evaluators):
            model.fit(self.dataset_train_extra_evaluators[i])
            y_pred = model.predict_value(self.dataset_val_extra_evaluators[i])
            self.active_learning_traj_extra_dict[i]['training_size'].append(self.train_size)
            for metric in self.metrics:
                metric_value = eval_metric_func(self.dataset_val_extra_evaluators[i].y, y_pred, metric=metric)
                self.active_learning_traj_extra_dict[i][metric].append(metric_value)
            pd.DataFrame(self.active_learning_traj_extra_dict[i]).to_csv(
                '%s/active_learning_extra_%d.traj' % (self.save_dir, i), index=False)

    def add_samples(self):
        pool_idx = list(range(self.pool_size))
        if self.learning_type == 'explorative':
            y_std = self.model_selector.predict_uncertainty(self.dataset_pool_selector)
            self.acquisition = y_std[np.argsort(y_std)[-self.batch_size:]].tolist()
            self.info('add a sample with acquisition: %s' % json.dumps(self.acquisition))
            selected_idx = self.get_selected_idx(y_std, pool_idx)
        elif self.learning_type == 'exploitive':
            y_pred = self.model_selector.predict_value(self.dataset_pool_selector)
            self.acquisition = y_pred[np.argsort(y_pred)[-self.batch_size:]].tolist()
            self.info('add a sample with acquisition: %s' % json.dumps(self.acquisition))
            selected_idx = self.get_selected_idx(y_pred, pool_idx)
        elif self.learning_type == 'EI':
            # TODO
            return
        elif self.learning_type == 'passive':
            if self.pool_size <= self.batch_size:
                selected_idx = pool_idx
            else:
                np.random.seed(self.seed)
                selected_idx = np.random.choice(pool_idx, self.batch_size, replace=False).tolist()
        else:
            raise ValueError(f'unknown learning type: {self.learning_type}')

        for i in sorted(selected_idx, reverse=True):
            self.dataset_train_selector.data.append(self.dataset_pool_selector.data.pop(i))
            if self.yoked_learning:
                self.dataset_train_evaluator.data.append(self.dataset_pool_evaluator.data.pop(i))
            for j in range(len(self.model_extra_evaluators)):
                self.dataset_train_extra_evaluators[j].data.append(self.dataset_pool_extra_evaluators[j].data.pop(i))

    def get_selected_idx(self,
                         acquisition_values: List[float],
                         pool_idx: List[int]) -> List[int]:
        # add all if the pool set is smaller than batch size.
        if len(acquisition_values) < self.batch_size:
            return pool_idx
        elif self.batch_algorithm == 'cluster':
            cluster_idx = self.get_nlargest_idx(acquisition_values, pool_idx, n_selected=self.cluster_size)
            K = self.kernel(self.dataset_pool_selector.X_for_kernel[np.asarray(cluster_idx)])
            add_idx = self.find_distant_samples(K)
            return np.array(cluster_idx)[add_idx]
        elif self.batch_algorithm == 'nlargest':
            return self.get_nlargest_idx(acquisition_values, pool_idx, n_selected=self.batch_size)
        else:
            raise ValueError(f'unknown batch_algorithm: {self.batch_algorithm}')

    @staticmethod
    def get_nlargest_idx(acquisition_values: List[float],
                         pool_idx: List[int],
                         n_selected: int) -> List[int]:
        if n_selected == 0 or len(acquisition_values) < n_selected:
            return pool_idx
        else:
            return np.asarray(pool_idx)[np.argsort(acquisition_values)[-n_selected:]].tolist()

    def find_distant_samples(self, gram_matrix: List[List[float]]) -> List[int]:
        """Find distant samples from a pool using clustering method.

        Parameters
        ----------
        gram_matrix: gram matrix of the samples.

        Returns
        -------
        List of idx
        """
        embedding = SpectralEmbedding(
            n_components=self.batch_size,
            affinity='precomputed'
        ).fit_transform(gram_matrix)

        cluster_result = KMeans(
            n_clusters=self.batch_size,
            # random_state=self.args.seed
        ).fit_predict(embedding)
        # find all center of clustering
        center = np.array([embedding[cluster_result == i].mean(axis=0)
                           for i in range(self.batch_size)])
        total_distance = defaultdict(
            dict)  # (key: cluster_idx, val: dict of (key:sum of distance, val:idx))
        for i in range(len(cluster_result)):
            cluster_class = cluster_result[i]
            total_distance[cluster_class][((np.square(
                embedding[i] - np.delete(center, cluster_class, axis=0))).sum(
                axis=1) ** -0.5).sum()] = i
        add_idx = [total_distance[i][min(total_distance[i].keys())] for i in
                   range(
                       self.batch_size)]  # find min-in-cluster-distance associated idx
        return add_idx

    def save(self, path, filename='al.pkl', overwrite=False):
        f_al = os.path.join(path, filename)
        if os.path.isfile(f_al) and not overwrite:
            raise RuntimeError(
                f'Path {f_al} already exists. To overwrite, set '
                '`overwrite=True`.'
            )
        store = self.__dict__.copy()
        pickle.dump(store, open(f_al, 'wb'), protocol=4)

    @classmethod
    def load(cls, path, filename='al.pkl'):
        f_al = os.path.join(path, filename)
        store = pickle.load(open(f_al, 'rb'))
        input = {}
        for key in ['save_dir', 'dataset_type', 'metrics', 'learning_type', 'model_selector', 'dataset_train_selector',
                    'dataset_pool_selector', 'dataset_val_evaluator']:
            input[key] = store[key]
        dataset = cls(**input)
        dataset.__dict__.update(**store)
        return dataset

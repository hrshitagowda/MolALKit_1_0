#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple, Callable
import os
import json
import pickle
import shutil
import pandas as pd
import numpy as np
from logging import Logger
from sklearn.metrics import *
from ..args import Metric
from .selection_method import BaseSelectionMethod, RandomSelectionMethod
from .forgetter import BaseForgetter, RandomForgetter, FirstForgetter


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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ActiveLearningResult:
    def __init__(self, n_iter: int):
        self.n_iter = n_iter
        self.results = dict()


class ActiveLearningTrajectory:
    def __init__(self, metrics: List[str] = None, top_k_score: bool = False):
        self.metrics = metrics
        self.top_k_score = top_k_score
        self.results = []

    def is_valid_column(self, column_name: str) -> Tuple[bool, List]:
        results = [alr.results.get(column_name) for alr in self.results]
        for result in results:
            if isinstance(result, float):
                return True, results
        return False, results

    def get_results(self) -> Dict:
        results_dict = dict()
        results_dict['n_iter'] = [alr.n_iter for alr in self.results]
        if self.metrics is not None:
            for metric in self.metrics:
                is_valid, results = self.is_valid_column(f'{metric}_selector')
                if is_valid:
                    results_dict[f'{metric}_selector'] = results
                for i in range(100):
                    is_valid, results = self.is_valid_column(f'{metric}_evaluator_{i}')
                    if is_valid:
                        results_dict[f'{metric}_evaluator_{i}'] = results
                    else:
                        break
        if self.top_k_score:
            is_valid, results = self.is_valid_column('top_k_score')
            if is_valid:
                results_dict['top_k_score'] = results
        results_dict['id_add'] = [json.dumps(alr.id_add, cls=NpEncoder) for alr in self.results]
        results_dict['acquisition_add'] = [json.dumps(alr.acquisition_add, cls=NpEncoder) for alr in self.results]
        results_dict['id_forgotten'] = [json.dumps(alr.id_forgotten, cls=NpEncoder) for alr in self.results]
        results_dict['acquisition_forgotten'] = [json.dumps(alr.acquisition_forgotten,
                                                            cls=NpEncoder) for alr in self.results]
        results_dict['id_prior_al'] = [json.dumps(alr.id_prior_al, cls=NpEncoder) for alr in self.results]
        return results_dict


class ActiveLearner:
    def __init__(self, save_dir: str, selection_method: BaseSelectionMethod, forgetter: BaseForgetter,
                 model_selector, dataset_train_selector, dataset_pool_selector, dataset_val_selector,
                 metrics: List[Metric], top_k_id: List[int] = None,
                 model_evaluators=None, dataset_train_evaluators=None,
                 dataset_pool_evaluators=None, dataset_val_evaluators=None,
                 yoked_learning_only: bool = False,
                 stop_size: int = None, evaluate_stride: int = None, kernel: Callable = None,
                 save_cpt_stride: int = None,
                 seed: int = 0,
                 logger: Logger = None):
        self.save_dir = save_dir
        self.selection_method = selection_method
        self.forgetter = forgetter
        self.model_selector = model_selector
        self.dataset_train_selector = dataset_train_selector
        self.dataset_pool_selector = dataset_pool_selector
        self.dataset_val_selector = dataset_val_selector
        self.metrics = metrics
        self.top_k_id = top_k_id
        self.model_evaluators = model_evaluators or []
        self.dataset_train_evaluators = dataset_train_evaluators or []
        self.dataset_pool_evaluators = dataset_pool_evaluators or []
        self.dataset_val_evaluators = dataset_val_evaluators or []
        self.yoked_learning_only = yoked_learning_only

        self.stop_size = stop_size
        self.evaluate_stride = evaluate_stride
        self.kernel = kernel  # used for cluster selection method
        self.save_cpt_stride = save_cpt_stride

        self.seed = seed
        if logger is not None:
            self.info = logger.info
        else:
            self.info = print
        self.current_iter = 0
        self.active_learning_traj = ActiveLearningTrajectory(metrics=self.metrics,
                                                             top_k_score=self.top_k_id is not None)

    @property
    def train_size(self) -> int:
        return len(self.dataset_train_selector)

    @property
    def pool_size(self) -> int:
        return len(self.dataset_pool_selector)

    def termination(self) -> bool:
        if self.pool_size == 0:
            self.info('Terminating active learning: pool size = 0')
            return True
        elif self.stop_size is not None and self.train_size >= self.stop_size:
            self.info(f'Terminating active learning: train_size > stop_size {self.stop_size}')
            return True
        else:
            return False

    def run(self, max_iter: int = None):
        info = f'Start active learning with training set size = {self.train_size}, pool set size = {self.pool_size}.' \
               f'\nSelection method: {self.selection_method.info}.'
        if self.forgetter is not None:
            info += f'\nForgetter: {self.forgetter.info}.'
        self.info(info)
        self.model_fitted = False
        for n_iter in range(self.current_iter, max_iter):
            alr = ActiveLearningResult(n_iter)
            alr.id_prior_al = self.get_id(self.dataset_train_selector)
            if self.termination():
                break
            self.info('Start an new iteration of active learning: %d.' % self.current_iter)
            # evaluate
            if self.evaluate_stride is not None and n_iter % self.evaluate_stride == 0:
                self.evaluate(alr)
            # add sample
            self.add_samples(alr)
            # forget sample
            self.forget_samples(alr)
            # save checkpoint file
            if self.save_cpt_stride is not None and n_iter % self.save_cpt_stride == 0:
                self.save(path=self.save_dir, filename='al_temp.pkl', overwrite=True)
                shutil.move(os.path.join(self.save_dir, 'al_temp.pkl'), os.path.join(self.save_dir, 'al.pkl'))
                self.info('save checkpoint file %s/al.pkl' % self.save_dir)

            self.current_iter += 1
            self.info('Training set size = %i' % self.train_size)
            self.info('Pool set size = %i' % self.pool_size)
            self.active_learning_traj.results.append(alr)

        df_traj = pd.DataFrame(self.active_learning_traj.get_results())
        df_traj.to_csv(os.path.join(self.save_dir, 'al_traj.csv'), index=False)
        if self.save_cpt_stride:
            self.save(path=self.save_dir, overwrite=True)

    def evaluate(self, alr: ActiveLearningResult):
        self.info('evaluating model performance.')
        # conventional active learning.
        if not self.yoked_learning_only:
            # evaluate the prediction performance of ML model on the validation set
            if self.metrics is not None:
                if not self.model_fitted:
                    self.model_selector.fit_alb(self.dataset_train_selector)
                    self.model_fitted = True
                y_pred = self.model_selector.predict_value(self.dataset_val_selector)
                for metric in self.metrics:
                    metric_value = eval_metric_func(self.dataset_val_selector.y, y_pred, metric=metric)
                    alr.results[f'{metric}_selector'] = metric_value
            # evaluate the percentage of top-k data selected in the training set
            if self.top_k_id is not None:
                alr.results['top_k_score'] = self.get_top_k_score(self.dataset_train_selector, self.top_k_id)
        # yoked learning
        for i, model in enumerate(self.model_evaluators):
            if self.metrics is not None:
                model.fit_alb(self.dataset_train_evaluators[i])
                y_pred = model.predict_value(self.dataset_val_evaluators[i])
                for metric in self.metrics:
                    metric_value = eval_metric_func(self.dataset_val_evaluators[i].y, y_pred, metric=metric)
                    alr.results[f'{metric}_evaluator_{i}'] = metric_value
        self.info('evaluating model performance finished.')

    def add_samples(self, alr: ActiveLearningResult):
        # train the model if it is not trained in the evaluation step, and the selection method is not random.
        if not self.model_fitted and not isinstance(self.selection_method, RandomSelectionMethod):
            self.model_selector.fit_alb(self.dataset_train_selector)
        selected_idx, acquisition = self.selection_method(model=self.model_selector,
                                                          data_train=self.dataset_train_selector,
                                                          data_pool=self.dataset_pool_selector,
                                                          kernel=self.kernel)
        alr.id_add = [self.dataset_pool_selector.data[i].id for i in selected_idx]
        alr.acquisition_add = acquisition
        # transfer data from pool to train.
        for i in sorted(selected_idx, reverse=True):
            self.dataset_train_selector.data.append(self.dataset_pool_selector.data.pop(i))
            for j in range(len(self.model_evaluators)):
                self.dataset_train_evaluators[j].data.append(self.dataset_pool_evaluators[j].data.pop(i))
        # set the model unfitted because new data is added.
        self.model_fitted = False

    def forget_samples(self, alr: ActiveLearningResult):
        # no forget algorithm is applied.
        # or forget algorithm is applied when self.forgetter.forget_size <= self.train_size.
        if self.forgetter is None or (self.forgetter.forget_size is not None and
                                      self.forgetter.forget_size > self.train_size):
            alr.id_forgotten = []
            alr.acquisition_forgotten = []
            return
        # train the model if the forgetter is not random or first.
        if not self.model_fitted and not self.forgetter.__class__ in [RandomForgetter, FirstForgetter]:
            self.model_selector.fit_alb(self.dataset_train_selector)
        # forget algorithm is applied.
        forgotten_idx, acquisition = self.forgetter(model=self.model_selector,
                                                    data=self.dataset_train_selector,
                                                    batch_size=self.forgetter.batch_size,
                                                    cutoff=self.forgetter.forget_cutoff)
        if forgotten_idx:
            alr.id_forgotten = [self.dataset_train_selector.data[i].id for i in forgotten_idx]
            alr.acquisition_forgotten = acquisition
            # transfer data from train to pool.
            for i in sorted(forgotten_idx, reverse=True):
                self.dataset_pool_selector.data.append(self.dataset_train_selector.data.pop(i))
                for j in range(len(self.model_evaluators)):
                    self.dataset_pool_evaluators[j].data.append(self.dataset_train_evaluators[j].data.pop(i))
        else:
            alr.id_forgotten = []
            alr.acquisition_forgotten = []

    @staticmethod
    def get_id(dataset):
        id_list = []
        for data in dataset:
            id_list.append(data.id)
        return id_list

    @staticmethod
    def get_top_k_score(dataset, top_k_id) -> float:
        N_top_k = 0
        for data in dataset:
            if data.id in top_k_id:
                N_top_k += 1
        return N_top_k / len(top_k_id)

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
                    'dataset_pool_selector', 'dataset_val_evaluator', 'df_train', 'df_pool']:
            input[key] = store[key]
        dataset = cls(**input)
        dataset.__dict__.update(**store)
        return dataset

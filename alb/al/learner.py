#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import json
import pickle
import shutil
import pandas as pd
import numpy as np
import math
import random
import scipy.stats as stats
from scipy.signal import savgol_filter
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
                 df_train,
                 df_pool,
                 batch_size: int = 1,
                 batch_algorithm: Literal['nlargest', 'cluster'] = 'nlargest',
                 stop_size=None,
                 kernel=None,
                 cluster_size=None,
                 forget_iter=None,
                 forget_protocol=None,
                 forget_size=None,
                 forget_percent=None,
                 error_rate=None,
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
        self.forget_iter = forget_iter
        self.forget_protocol = forget_protocol
        self.forget_size = forget_size
        self.forget_percent = forget_percent
        self.error_rate = error_rate
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

        self.df_train = df_train
        self.df_train['acquisition'] = 'none'
        self.df_pool = df_pool

        self.evaluate_stride = evaluate_stride
        self.extra_evaluators_only = extra_evaluators_only
        self.save_cpt_stride = save_cpt_stride
        
        self.threshold = 0.3
        self.window = 100

        self.seed = seed
        if logger is not None:
            self.info = logger.info
        else:
            self.info = print
        self.n_iter = 0
        self.n_forgotten_data = 0
        self.selected_data = dataset_train_selector.repr.ravel().tolist()

    @property
    def active_learning_traj_dict(self) -> Dict:
        if not hasattr(self, '_active_learning_traj_dict'):
            self._active_learning_traj_dict = {'iteration': [],
                                               'selected_data': [],
                                               'forgotten_data': [],
                                               'training_size': [],
                                               'acquisition': []}
            for metric in self.metrics:
                self._active_learning_traj_dict[metric] = []
        return self._active_learning_traj_dict

    @property
    def active_learning_traj_extra_dict(self) -> List[Dict]:
        if not hasattr(self, '_active_learning_traj_extra_dict'):
            self._active_learning_traj_extra_dict = [{'iteration': [],
                                                      'selected_data': [],
                                                      'forgotten_data': [],
                                                      'training_size': [],
                                                      'acquisition': []} for i in
                                                     range(len(self.model_extra_evaluators))]
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
            self.info('Terminating active learning: pool size = 0')
            return True
        elif self.stop_size is not None and self.train_size >= self.stop_size:
            self.info('Terminating active learning: train size > stop size' % self.stop_size)
            return True
        else:
            return False
        
    def window_ttest(self, maxWindow, currentWindow, alt):
        #perform one-sided t-test between current window and maximum window
        return stats.ttest_ind(maxWindow, currentWindow, alternative=alt)[1]
    
    def find_maxIter(self, i, data, window, maxIter):
        #protocol for determining the maximum iteration, which corresponds to the time to begin forgetting at
        
        #filter data using Savitzky-Golay filter
        filter_data = savgol_filter(data, 50, 3)
        #define current window
        currentWindow = filter_data[-window:]
        #define maximum performance window
        maxWindow = filter_data[maxIter-window+1:maxIter+1]
        #ttest if max window is significantly greater than current window
        pval_greater = self.window_ttest(maxWindow, currentWindow, 'greater')
        #ttest if max window is significantly less than current window
        pval_less = self.window_ttest(maxWindow, currentWindow, 'less')
        if pval_less < 0.001:
            #assign new maximum iteration if maximum window is significantly less than current window
            #threshold experimentally determined
            self.info('Found new maximum iteration')
            maxIter = i
        elif pval_greater < 0.01:
            #return previously found maximum iteration if maximum window is significantly greater
            #adjust to get highest point of performance within window
            maxIter = maxIter + (np.argmax(data[maxIter-window:maxIter])-window)
            self.info('Maximum iteration found, returning to iteration: %d.' % maxIter)
            return True, maxIter
        
        return False, maxIter
    
    def reset_to_max(self, i, maxIter):
        #return to the maximum iteration identified
        iter_diff = i - maxIter

        for j in range(iter_diff+1):
            #return datapoints added to train set since maxIter to pool set (but maintain selected data)
            self.dataset_pool_selector.data.append(self.dataset_train_selector.data.pop()) 
            #remove relevant trajectory entries
            for key, traj in self.active_learning_traj_dict.items():
                traj.pop()
            self.n_iter -= 1
            i -= 1
        
        self.info('Successfully returned to iteration: %d.' % self.n_iter)
        return i

    def run(self, n_iter: int = None):
        self.info('start active learning with training set size = %d' % self.train_size)
        self.info('pool set size = %d' % self.pool_size)
        self.info(f'implement forget protocol with {self.forget_iter}, {self.forget_protocol}, {self.forget_size}')
        total_size = self.train_size+self.pool_size

        if n_iter is None:
            n_iter = total_size
        
        #find_iter parameters
        run_forget = False
        threshold = int(self.threshold*total_size) #identify threshold value to begin searching at
        threshIter = threshold #initialize maximum iteration at threshold
        maxIter = threshIter
        al_max_found = False
        forget_max_found = False
        f_maxIter = 0
        
        i = 0
        while i < (n_iter):
        #for i in range(n_iter):
            if self.termination():
                break
            self.info('Start an new iteration of active learning: %d.' % self.n_iter)
            # training
            self.model_selector.fit(self.dataset_train_selector)
            # evaluate
            if self.evaluate_stride is not None and self.train_size % self.evaluate_stride == 0:
                self.evaluate()
            # add sample
            self.add_samples()
            
            # forget sample
            if self.forget_iter == 'find_iter':
                #find the iteration to begin forgetting at
                if run_forget == False:
                    perf_metric = 'mcc'
                    traj_data = np.array(self.active_learning_traj_dict[perf_metric])
                    #mcc trajectory (from test data)
                    if len(traj_data) >= threshold: 
                        #start searching after threshold
                        al_max_found, maxIter = self.find_maxIter(i, traj_data, self.window, maxIter)
                    if al_max_found == True:
                        i = self.reset_to_max(i, maxIter)
                        run_forget = True
                elif run_forget == True:
                    self.forget_samples()
            else:
                #standard forget protocol with set_iter
                self.forget_samples()
                    
            # save checkpoint file
            if self.save_cpt_stride is not None and i % self.save_cpt_stride == 0:
                self.save(path=self.save_dir, filename='al_temp.pkl', overwrite=True)
                shutil.move(os.path.join(self.save_dir, 'al_temp.pkl'), os.path.join(self.save_dir, 'al.pkl'))
                self.info('save checkpoint file %s/al.pkl' % self.save_dir)

            self.n_iter += 1
            i += 1
            self.info('Training set size = %i' % self.train_size)
            self.info('Pool set size = %i' % self.pool_size)

        if len(self.active_learning_traj_dict['training_size']) == 0 or \
                self.active_learning_traj_dict['training_size'][-1] != self.train_size:
            self.model_selector.fit(self.dataset_train_selector)
            self.evaluate()
        if self.save_cpt_stride:
            self.save(path=self.save_dir, overwrite=True)

    def evaluate(self):
        self.info('evaluating model performance.')
        if not self.extra_evaluators_only:
            if self.yoked_learning:
                self.model_evaluator.fit(self.dataset_train_evaluator)
            y_pred = self.model_evaluator.predict_value(self.dataset_val_evaluator)

            self.active_learning_traj_dict['iteration'].append(self.n_iter)
            self.active_learning_traj_dict['selected_data'].append(len(self.selected_data))
            self.active_learning_traj_dict['forgotten_data'].append(self.n_forgotten_data)
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
            self.active_learning_traj_extra_dict[i]['iteration'].append(self.n_iter)
            self.active_learning_traj_extra_dict[i]['selected_data'].append(len(self.selected_data))
            self.active_learning_traj_extra_dict[i]['forgotten_data'].append(self.n_forgotten_data)
            self.active_learning_traj_extra_dict[i]['training_size'].append(self.train_size)
            if hasattr(self, 'acquisition'):
                self.active_learning_traj_extra_dict[i]['acquisition'].append(json.dumps(self.acquisition))
            else:
                self.active_learning_traj_extra_dict[i]['acquisition'].append('none')
            for metric in self.metrics:
                metric_value = eval_metric_func(self.dataset_val_extra_evaluators[i].y, y_pred, metric=metric)
                self.active_learning_traj_extra_dict[i][metric].append(metric_value)
            pd.DataFrame(self.active_learning_traj_extra_dict[i]).to_csv(
                '%s/active_learning_extra_%d.traj' % (self.save_dir, i), index=False)

    def add_samples(self):
        pool_idx = list(range(self.pool_size))
        if self.learning_type == 'explorative':
            y_pred = self.model_selector.predict_value(self.dataset_pool_selector)
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

        df_add = self.df_pool[self.df_pool.index.isin(selected_idx)].reset_index().drop(columns=['index'])
        if hasattr(self, 'acquisition'):
            df_add['acquisition'] = self.acquisition
        else:
            df_add['acquisition'] = 'none'
        self.df_train = pd.concat([self.df_train, df_add]).reset_index().drop(columns=['index'])
        self.df_pool = self.df_pool[~self.df_pool.index.isin(selected_idx)].reset_index().drop(columns=['index'])
        self.df_train.to_csv('%s/train_al.csv' % self.save_dir, index=False)
        self.df_pool.to_csv('%s/pool_al.csv' % self.save_dir, index=False)

        for i in sorted(selected_idx, reverse=True):
            self.dataset_train_selector.data.append(self.dataset_pool_selector.data.pop(i))

            repr = self.dataset_train_selector.repr.ravel()[-1]
            if repr not in self.selected_data:
                self.selected_data.append(repr)

            if self.yoked_learning:
                self.dataset_train_evaluator.data.append(self.dataset_pool_evaluator.data.pop(i))
            for j in range(len(self.model_extra_evaluators)):
                self.dataset_train_extra_evaluators[j].data.append(self.dataset_pool_extra_evaluators[j].data.pop(i))

    def forget_first(self):
        #forget the first data point in the training set
        self.info('forgetting first sample in training set')
        self.dataset_pool_selector.data.append(self.dataset_train_selector.data.pop(0))
        self.n_forgotten_data += 1
        if self.yoked_learning:
            self.dataset_pool_evaluator.data.append(self.dataset_train_evaluator.data.pop(0))
        for j in range(len(self.model_extra_evaluators)):
            self.dataset_pool_extra_evaluators[j].data.append(self.dataset_train_extra_evaluators[j].data.pop(0))
            
    def forget_oob_least_uncertain(self):
        #forget the least uncertain datapoint in the training set based on the oob decision function 
        self.info('forgetting oob least uncertain sample in training set')
        y_oob_proba = self.model_selector.oob_decision_function_
        #uncertainty calculation
        y_oob_uncertainty = 0.25 - np.var(y_oob_proba, axis=1)
        #select the point with least uncertainty
        forget_ind = np.argmin(y_oob_uncertainty)
        self.dataset_pool_selector.data.append(self.dataset_train_selector.data.pop(forget_ind))
        self.n_forgotten_data += 1
        #yoked learning, still forget first
        if self.yoked_learning:
            self.dataset_pool_evaluator.data.append(self.dataset_train_evaluator.data.pop(forget_ind))
        for j in range(len(self.model_extra_evaluators)):
            self.dataset_pool_extra_evaluators[j].data.append(self.dataset_train_extra_evaluators[j].data.pop(forget_ind))
            
    def forget_oob_most_uncertain(self):
        #forget the most uncertain datapoint in the training set based on the oob decision function 
        self.info('forgetting oob most uncertain sample in training set')
        y_oob_proba = self.model_selector.oob_decision_function_
        #uncertainty calculation
        y_oob_uncertainty = 0.25 - np.var(y_oob_proba, axis=1)
        #select the point with most uncertainty
        forget_ind = np.argmax(y_oob_uncertainty)
        self.dataset_pool_selector.data.append(self.dataset_train_selector.data.pop(forget_ind))
        self.n_forgotten_data += 1
        #yoked learning, still forget first
        if self.yoked_learning:
            self.dataset_pool_evaluator.data.append(self.dataset_train_evaluator.data.pop(forget_ind))
        for j in range(len(self.model_extra_evaluators)):
            self.dataset_pool_extra_evaluators[j].data.append(self.dataset_train_extra_evaluators[j].data.pop(forget_ind))
            
    def forget_random(self):
        #forget the most uncertain datapoint in the training set based on the oob decision function 
        forget_ind = random.randrange(self.train_size)
        self.info(f'forgetting random sample ({forget_ind}) in training set ({self.train_size})')
        self.dataset_pool_selector.data.append(self.dataset_train_selector.data.pop(forget_ind))
        self.n_forgotten_data += 1
        #yoked learning, still forget first
        if self.yoked_learning:
            self.dataset_pool_evaluator.data.append(self.dataset_train_evaluator.data.pop(forget_ind))
        for j in range(len(self.model_extra_evaluators)):
            self.dataset_pool_extra_evaluators[j].data.append(self.dataset_train_extra_evaluators[j].data.pop(forget_ind))
                
    def forget_sample(self):
    #decide which protocol to implement based on input
        if self.forget_protocol == 'forget_first':
            self.forget_first()
        elif self.forget_protocol == 'oob_least_uncertain':
            self.forget_oob_least_uncertain()
        elif self.forget_protocol == 'oob_most_uncertain':
            self.forget_oob_most_uncertain()
        elif self.forget_protocol == 'forget_random':
            self.forget_random()
    
    def forget_samples(self):
    #forget a sample from training set
        dataset_size=self.train_size+self.pool_size
         
        if self.forget_iter == 'set_iter':
            #preset value to start forgetting at
            forget_iter = 0
            if self.forget_size is None and self.forget_percent is not None:
            #sliding window method, set percentage of training set
                forget_iter = math.floor(self.forget_percent/100*dataset_size)
            else:
            #specific forget size from maxIter
                forget_iter = self.forget_size
            if self.n_iter >= forget_iter:
            #forget sample
                self.info(f'forgetting at iteration {self.n_iter}, forget_iter: {forget_iter}')
                self.forget_sample()
        elif self.forget_iter == 'find_iter':
            #if finding iteration to begin forgetting at
            #for find_iter this method will only be called once the maximum has been found, so start forgetting by default
            self.info(f'forgetting at iteration {self.n_iter}')
            self.forget_sample()
        else:
            return
            
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
                    'dataset_pool_selector', 'dataset_val_evaluator', 'df_train', 'df_pool']:
            input[key] = store[key]
        dataset = cls(**input)
        dataset.__dict__.update(**store)
        return dataset

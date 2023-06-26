#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil

CWD = os.path.dirname(os.path.abspath(__file__))
from tap import Tap
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from logging import Logger
import json
import math
import pandas as pd
import numpy as np
from mgktools.features_mol import FeaturesGenerator
from mgktools.data.split import data_split_index
from alb.logging import create_logger
from alb.utils import get_data, get_model, get_kernel


Metric = Literal['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc',
                 'rmse', 'mae', 'mse', 'r2', 'max']


class CommonArgs(Tap):
    save_dir: str
    """the output directory."""
    n_jobs: int = 1
    """the cpu numbers used for parallel computing."""
    quiet: bool = False
    """Whether the stream handler should be quiet (i.e., print only important info)."""
    logger_name: str = 'alb_output'
    """the prefix of the output logger file: verbose.log and quite.log"""
    seed: int = 0
    """random seed."""

    def process_args(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = create_logger(self.logger_name, save_dir=self.save_dir, quiet=self.quiet)


class DatasetArgs(CommonArgs):
    data_public = None
    """Use public data sets."""
    data_path: str = None
    """the Path of input data CSV file."""
    data_path_training: str = None
    """the Path of input data CSV file for training set."""
    data_path_pool: str = None
    """the Path of input data CSV file for pool set."""
    data_path_val: str = None
    """the Path of input data CSV file for validation set."""
    pure_columns: List[str] = None
    """
    For pure compounds.
    Name of the columns containing single SMILES or InChI string.
    """
    mixture_columns: List[str] = None
    """
    For mixtures.
    Name of the columns containing multiple SMILES or InChI string and 
    corresponding concentration.
    example: ['C', 0.5, 'CC', 0.3]
    """
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    """
    feature_columns: List[str] = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    dataset_type: Literal['regression', 'classification', 'multiclass'] = None
    """
    Type of task.
    """
    split_type: Literal['random', 'scaffold_random', 'scaffold_order'] = None
    """Method of splitting the data into active learning/validation."""
    split_sizes: List[float] = None
    """Split proportions for active learning/validation sets."""
    full_val: bool = False
    """validate the performance of active learning on the full dataset."""
    error_rate: int = None
    """% of error to be introduced to the training set"""

    def process_args(self) -> None:
        super().process_args()
        if self.data_public == 'freesolv':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['freesolv']
            self.dataset_type = 'regression'
        elif self.data_public == 'delaney':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['logSolubility']
            self.dataset_type = 'regression'
        elif self.data_public == 'lipo':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['lipo']
            self.dataset_type = 'regression'
        elif self.data_public == 'pdbbind_refined':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['-logKd/Ki']
            self.dataset_type = 'regression'
        elif self.data_public == 'pdbbind_full':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['-logKd/Ki']
            self.dataset_type = 'regression'
        elif self.data_public in ['ld50_zhu', 'caco2_wang', 'solubility_aqsoldb', 'ppbr_az', 'vdss_lombardo',
                                  'Half_Life_Obach', 'Clearance_Hepatocyte_AZ']:
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['Drug']
            self.target_columns = ['Y']
            self.dataset_type = 'regression'
        elif self.data_public == 'bbbp':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['p_np']
            self.dataset_type = 'classification'
        elif self.data_public == 'bace':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['mol']
            self.target_columns = ['Class']
            self.dataset_type = 'classification'
        elif self.data_public == 'hiv':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['HIV_active']
            self.dataset_type = 'classification'
        elif self.data_public in ['ames', 'carcinogens_lagunin', 'dili', 'herg', 'skin', 'hia_hou', 'pgp_broccatelli',
                                  'bioavailability_ma', 'clintox', 'bbb_martins', 'CYP1A2_Veith',
                                  'CYP2C9_Substrate_CarbonMangels', 'CYP2C9_Veith', 'CYP2C19_Veith',
                                  'CYP2D6_Substrate_CarbonMangels', 'CYP2D6_Veith', 'CYP3A4_Veith',
                                  'CYP3A4_Substrate_CarbonMangels']:
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['Drug']
            self.target_columns = ['Y']
            self.dataset_type = 'classification'

        if self.split_type == 'scaffold':
            assert len(self.pure_columns) == 1
            assert self.mixture_columns is None

        if self.dataset_type != 'regression':
            assert len(self.target_columns) == 1

        if self.data_path is not None:
            assert self.data_path_val is None and self.data_path_training is None and self.data_path_pool is None
            df = pd.read_csv(self.data_path)
            if self.full_val:
                assert self.split_type == None
                assert self.split_sizes == None
                df.to_csv('%s/val.csv' % self.save_dir, index=False)
                df_al = df
                if self.dataset_type == 'regression':
                    train_index, pool_index = data_split_index(
                        n_samples=len(df_al),
                        mols=df_al[self.pure_columns[0]] if self.pure_columns is not None else None,
                        split_type='random',
                        sizes=[self.init_size / len(df_al), 1 - self.init_size / len(df_al)],
                        seed=self.seed)
                else:
                    train_index, pool_index = data_split_index(
                        n_samples=len(df_al),
                        mols=df_al[self.pure_columns[0]] if self.pure_columns is not None else None,
                        targets=df_al[self.target_columns[0]],
                        split_type='init_al',
                        n_samples_per_class=1,
                        seed=self.seed)
                    if self.init_size > 2:
                        train_index.extend(np.random.choice(pool_index, self.init_size - 2, replace=False))
                        _ = []
                        for i in pool_index:
                            if i not in train_index:
                                _.append(i)
                        pool_index = _
            else:
                al_index, val_index = data_split_index(
                    n_samples=len(df),
                    mols=df[self.pure_columns[0]] if self.pure_columns is not None else None,
                    # targets=df[self.target_columns[0]],
                    split_type=self.split_type,
                    sizes=self.split_sizes,
                    seed=self.seed,
                    logger=self.logger)
                df[df.index.isin(val_index)].to_csv('%s/val.csv' % self.save_dir, index=False)
                df_al = df[df.index.isin(al_index)]
                if self.init_size > len(df_al):
                    self.init_size = len(df_al)
                if self.dataset_type == 'regression':
                    train_index, pool_index = data_split_index(
                        n_samples=len(df_al),
                        mols=df_al[self.pure_columns[0]] if self.pure_columns is not None else None,
                        split_type='random',
                        sizes=[self.init_size / len(df_al), 1 - self.init_size / len(df_al)],
                        seed=self.seed)
                else:
                #classification task training set, split into train and pool sets
                    #introduce error to training set
                    if self.error_rate != 0 and self.error_rate is not None:
                        #numpy array of target values
                        targets_error=df_al[self.target_columns[0]].to_numpy(dtype=int,copy=True)
                        #get random indices for introducing error
                        n_train_samples=len(targets_error)
                        #generate same indices for introducing error
                        np.random.seed(self.seed)
                        error_index = np.random.choice(n_train_samples,int(self.error_rate/100*n_train_samples),replace=False)
                        #flip label
                        targets_error[error_index] ^= 1
                        #update target values with error
                        df_al[self.target_columns[0]] = targets_error
                    train_index, pool_index = data_split_index(
                        n_samples=len(df_al),
                        mols=df_al[self.pure_columns[0]] if self.pure_columns is not None else None,
                        targets=df_al[self.target_columns[0]],
                        split_type='init_al',
                        n_samples_per_class=1,
                        seed=self.seed)
                    if self.init_size > 2:
                        train_index.extend(np.random.choice(pool_index, self.init_size - 2, replace=False))
                        _ = []
                        for i in pool_index:
                            if i not in train_index:
                                _.append(i)
                        pool_index = _
            df_al.iloc[train_index].to_csv('%s/train_init.csv' % self.save_dir, index=False)
            df_al.iloc[pool_index].to_csv('%s/pool_init.csv' % self.save_dir, index=False)
        else:
            assert self.data_path_training is not None, 'please provide input data'
            assert self.data_path_pool is not None, 'please provide input data'
            assert self.data_path_val is not None, 'please provide input data'
            shutil.copyfile(self.data_path_training, '%s/train_init.csv' % self.save_dir)
            shutil.copyfile(self.data_path_pool, '%s/pool_init.csv' % self.save_dir)
            shutil.copyfile(self.data_path_val, '%s/val.csv' % self.save_dir)
            pd.concat([pd.read_csv(f) for f in [self.data_path_training,
                                                self.data_path_pool,
                                                self.data_path_val]]).to_csv('%s/full.csv' % self.save_dir)
            self.data_path = '%s/full.csv' % self.save_dir


class ModelArgs(Tap):
    model_config_selector: str
    """config file contain all information of the machine learning model."""
    model_config_evaluator: str = None
    """config file contain all information of the machine learning model for performance evaluation."""
    model_config_extra_evaluators: List[str] = None
    """A list of config files contain all information of the machine learning model for performance evaluation."""

    @property
    def yoked_learning(self) -> bool:
        if self.model_config_evaluator is None:
            return False
        else:
            return True

    @property
    def model_config_selector_dict(self) -> Dict:
        return json.loads(open(self.model_config_selector).read())

    @property
    def model_config_evaluator_dict(self) -> Dict:
        if not self.yoked_learning:
            return self.model_config_selector_dict
        else:
            return json.loads(open(self.model_config_evaluator).read())

    @property
    def model_config_extra_evaluators_dict(self) -> List[Dict]:
        if self.model_config_extra_evaluators is None:
            return []
        else:
            return [json.loads(open(m).read()) for m in self.model_config_extra_evaluators]


class ActiveLearningArgs(DatasetArgs, ModelArgs):
    save_dir: str
    """the output directory."""
    n_jobs: int = 1
    """the cpu numbers used for parallel computing."""
    data_path: str = None
    """the Path of input data CSV file."""
    learning_type: Literal['explorative', 'exploitive', 'EI', 'passive']
    """the learning type to be performed."""
    metrics: List[Metric]
    """the metrics to evaluate model performance."""
    evaluate_stride: int = 100
    """evaluate model performance on the validation set when the size of the training set is an integer multiple of the 
    evaluation stride."""
    extra_evaluators_only: bool = False
    """Output active learing trajectory of extra evaluators only."""
    init_size: int = 2
    """number of samples as the initial."""
    batch_size: int = 1
    """number of samples added in each active learning iteration."""
    batch_style: Literal['nlargest', 'clustering'] = 'nlargest'
    """the method that add a batch of samples."""
    stop_ratio: float = None
    """the ratio of molecules to stop the active learning."""
    stop_size: int = None
    """the number of molecules to stop the active learning."""
    forget_iter: str = None
    """when to start forgetting (find_iter, set_iter)."""
    forget_protocol: str = None
    """protocol to use (forget_first, oob_least_uncertain (RF only), oob_most_uncertain (RF only), forget_random)."""
    forget_size: int = None
    """the number of moleculesin the training set to start forgetting data at (use with set_iter)."""
    forget_percent: int = None
    """the percent of the full training set to start forgetting data (use with set_iter)."""
    error_rate: int = None
    """the percent of the training set that will be affected by error."""
    save_cpt_stride: int = None
    """save checkpoint file every no. steps of active learning iteration."""
    load_checkpoint: bool = False
    """load"""
    n_iter: int = None
    """number of iterations of active learning to performed, None means stop until all data are selected."""

    @property
    def model_selector(self):
        if not hasattr(self, '_model_selector'):
            self._model_selector = get_model(
                data_format=self.model_config_selector_dict['data_format'],
                dataset_type=self.dataset_type,
                model=self.model_config_selector_dict.get('model'),
                save_dir='%s/selector' % self.save_dir,
                loss_function=self.model_config_selector_dict.get('loss_function'),
                num_tasks=len(self.target_columns),
                multiclass_num_classes=self.model_config_selector_dict.get('loss_function') or 3,
                features_generator=self.features_generator_selector,
                no_features_scaling=self.model_config_selector_dict.get('no_features_scaling') or False,
                features_only=self.model_config_selector_dict.get('features_only') or False,
                features_size=self.data_train_selector.features_size(),
                epochs=self.model_config_selector_dict.get('epochs') or 30,
                depth=self.model_config_selector_dict.get('depth') or 3,
                hidden_size=self.model_config_selector_dict.get('hidden_size') or 300,
                ffn_num_layers=self.model_config_selector_dict.get('ffn_num_layers') or 2,
                ffn_hidden_size=self.model_config_selector_dict.get('ffn_hidden_size'),
                dropout=self.model_config_selector_dict.get('dropout') or 0.0,
                batch_size=self.model_config_selector_dict.get('batch_size') or 50,
                ensemble_size=self.model_config_selector_dict.get('ensemble_size') or 1,
                number_of_molecules=self.model_config_selector_dict.get('number_of_molecules') or 1,
                mpn_shared=self.model_config_selector_dict.get('mpn_shared') or False,
                atom_messages=self.model_config_selector_dict.get('atom_messages') or False,
                undirected=self.model_config_selector_dict.get('undirected') or False,
                class_balance=self.model_config_selector_dict.get('class_balance') or False,
                checkpoint_dir=self.model_config_selector_dict.get('checkpoint_dir'),
                checkpoint_frzn=self.model_config_selector_dict.get('checkpoint_frzn'),
                frzn_ffn_layers=self.model_config_selector_dict.get('frzn_ffn_layers') or 0,
                freeze_first_only=self.model_config_selector_dict.get('freeze_first_only') or False,
                kernel=self.kernel_selector,
                uncertainty_type=self.model_config_selector_dict.get('uncertainty_type'),
                alpha=self.model_config_selector_dict.get('alpha'),
                n_jobs=self.n_jobs,
                seed=self.seed,
                logger=self.logger
            )
        return self._model_selector

    @property
    def model_evaluator(self):
        if self.yoked_learning:
            if not hasattr(self, '_model_evaluator'):
                self._model_evaluator = get_model(
                    data_format=self.model_config_evaluator_dict['data_format'],
                    dataset_type=self.dataset_type,
                    model=self.model_config_evaluator_dict.get('model'),
                    save_dir='%s/evaluator' % self.save_dir,
                    loss_function=self.model_config_evaluator_dict.get('loss_function'),
                    num_tasks=len(self.target_columns),
                    multiclass_num_classes=self.model_config_evaluator_dict.get('loss_function') or 3,
                    features_generator=self.features_generator_evaluator,
                    no_features_scaling=self.model_config_evaluator_dict.get('no_features_scaling') or False,
                    features_only=self.model_config_evaluator_dict.get('features_only') or False,
                    features_size=self.data_train_evaluator.features_size(),
                    epochs=self.model_config_evaluator_dict.get('epochs') or 30,
                    depth=self.model_config_evaluator_dict.get('depth') or 3,
                    hidden_size=self.model_config_evaluator_dict.get('hidden_size') or 300,
                    ffn_num_layers=self.model_config_evaluator_dict.get('ffn_num_layers') or 2,
                    ffn_hidden_size=self.model_config_evaluator_dict.get('ffn_hidden_size'),
                    dropout=self.model_config_evaluator_dict.get('dropout') or 0.0,
                    batch_size=self.model_config_evaluator_dict.get('batch_size') or 50,
                    ensemble_size=self.model_config_evaluator_dict.get('ensemble_size') or 1,
                    number_of_molecules=self.model_config_evaluator_dict.get('number_of_molecules') or 1,
                    mpn_shared=self.model_config_evaluator_dict.get('mpn_shared') or False,
                    atom_messages=self.model_config_evaluator_dict.get('atom_messages') or False,
                    undirected=self.model_config_evaluator_dict.get('undirected') or False,
                    class_balance=self.model_config_evaluator_dict.get('class_balance') or False,
                    checkpoint_dir=self.model_config_evaluator_dict.get('checkpoint_dir'),
                    checkpoint_frzn=self.model_config_evaluator_dict.get('checkpoint_frzn'),
                    frzn_ffn_layers=self.model_config_evaluator_dict.get('frzn_ffn_layers') or 0,
                    freeze_first_only=self.model_config_evaluator_dict.get('freeze_first_only') or False,
                    kernel=self.kernel_evaluator,
                    uncertainty_type=self.model_config_evaluator_dict.get('uncertainty_type'),
                    alpha=self.model_config_evaluator_dict.get('alpha'),
                    n_jobs=self.n_jobs,
                    seed=self.seed,
                    logger=self.logger
                )
            return self._model_evaluator
        else:
            return self.model_selector

    @property
    def model_extra_evaluators(self):
        if not hasattr(self, '_model_extra_evaluators'):
            self._model_extra_evaluators = [get_model(
                data_format=model_config['data_format'],
                dataset_type=self.dataset_type,
                model=model_config.get('model'),
                save_dir='%s/extra_evaluator_%d' % (self.save_dir, i),
                loss_function=model_config.get('loss_function'),
                num_tasks=len(self.target_columns),
                multiclass_num_classes=model_config.get('loss_function') or 3,
                features_generator=self.features_generator_extra_evaluators[i],
                no_features_scaling=model_config.get('no_features_scaling') or False,
                features_only=model_config.get('features_only') or False,
                features_size=self.data_train_extra_evaluators[i].features_size(),
                epochs=model_config.get('epochs') or 30,
                depth=model_config.get('depth') or 3,
                hidden_size=model_config.get('hidden_size') or 300,
                ffn_num_layers=model_config.get('ffn_num_layers') or 2,
                ffn_hidden_size=model_config.get('ffn_hidden_size'),
                dropout=model_config.get('dropout') or 0.0,
                batch_size=model_config.get('batch_size') or 50,
                ensemble_size=model_config.get('ensemble_size') or 1,
                number_of_molecules=model_config.get('number_of_molecules') or 1,
                mpn_shared=model_config.get('mpn_shared') or False,
                atom_messages=model_config.get('atom_messages') or False,
                undirected=model_config.get('undirected') or False,
                class_balance=model_config.get('class_balance') or False,
                checkpoint_dir=model_config.get('checkpoint_dir'),
                checkpoint_frzn=model_config.get('checkpoint_frzn'),
                frzn_ffn_layers=model_config.get('frzn_ffn_layers') or 0,
                freeze_first_only=model_config.get('freeze_first_only') or False,
                kernel=self.kernel_extra_evaluators[i],
                uncertainty_type=model_config.get('uncertainty_type'),
                alpha=model_config.get('alpha'),
                n_jobs=self.n_jobs,
                seed=self.seed,
                logger=self.logger
            ) for i, model_config in enumerate(self.model_config_extra_evaluators_dict)]
        return self._model_extra_evaluators

    @property
    def data_train_selector(self):
        if not hasattr(self, '_data_train_selector'):
            self._data_train_selector = get_data(
                data_format=self.model_config_selector_dict['data_format'],
                path='%s/train_init.csv' % self.save_dir,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_selector,
                features_combination=self.model_config_selector_dict.get('features_combination'),
                graph_kernel_type=self.model_config_selector_dict.get('graph_kernel_type'),
                n_jobs=self.n_jobs)
        return self._data_train_selector

    @property
    def data_pool_selector(self):
        if not hasattr(self, '_data_pool_selector'):
            self._data_pool_selector = get_data(
                data_format=self.model_config_selector_dict['data_format'],
                path='%s/pool_init.csv' % self.save_dir,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_selector,
                features_combination=self.model_config_selector_dict.get('features_combination'),
                graph_kernel_type=self.model_config_selector_dict.get('graph_kernel_type'),
                n_jobs=self.n_jobs)
        return self._data_pool_selector

    @property
    def data_train_evaluator(self):
        if self.yoked_learning:
            if not hasattr(self, '_data_train_evaluator'):
                self._data_train_evaluator = get_data(
                    data_format=self.model_config_evaluator_dict['data_format'],
                    path='%s/train_init.csv' % self.save_dir,
                    pure_columns=self.pure_columns,
                    mixture_columns=self.mixture_columns,
                    target_columns=self.target_columns,
                    feature_columns=self.feature_columns,
                    features_generator=self.features_generator_evaluator,
                    features_combination=self.model_config_evaluator_dict.get('features_combination'),
                    graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
                    n_jobs=self.n_jobs)
            return self._data_train_evaluator
        else:
            return self.data_train_selector

    @property
    def data_pool_evaluator(self):
        if self.yoked_learning:
            if not hasattr(self, '_data_pool_evaluator'):
                self._data_pool_evaluator = get_data(
                    data_format=self.model_config_evaluator_dict['data_format'],
                    path='%s/pool_init.csv' % self.save_dir,
                    pure_columns=self.pure_columns,
                    mixture_columns=self.mixture_columns,
                    target_columns=self.target_columns,
                    feature_columns=self.feature_columns,
                    features_generator=self.features_generator_evaluator,
                    features_combination=self.model_config_evaluator_dict.get('features_combination'),
                    graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
                    n_jobs=self.n_jobs)
            return self._data_pool_evaluator
        else:
            return self.data_pool_selector

    @property
    def data_val_evaluator(self):
        if not hasattr(self, '_data_val_evaluator'):
            self._data_val_evaluator = get_data(
                data_format=self.model_config_evaluator_dict['data_format'],
                path='%s/val.csv' % self.save_dir,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_evaluator,
                features_combination=self.model_config_evaluator_dict.get('features_combination'),
                graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
                n_jobs=self.n_jobs)
        return self._data_val_evaluator

    @property
    def data_train_extra_evaluators(self):
        if not hasattr(self, '_data_train_extra_evaluators'):
            self._data_train_extra_evaluators = [get_data(
                data_format=model_config['data_format'],
                path='%s/train_init.csv' % self.save_dir,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_extra_evaluators[i],
                features_combination=model_config.get('features_combination'),
                graph_kernel_type=model_config.get('graph_kernel_type'),
                n_jobs=self.n_jobs) for i, model_config in enumerate(self.model_config_extra_evaluators_dict)]
        return self._data_train_extra_evaluators

    @property
    def data_pool_extra_evaluators(self):
        if not hasattr(self, '_data_pool_extra_evaluators'):
            self._data_pool_extra_evaluators = [get_data(
                data_format=model_config['data_format'],
                path='%s/pool_init.csv' % self.save_dir,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_extra_evaluators[i],
                features_combination=model_config.get('features_combination'),
                graph_kernel_type=model_config.get('graph_kernel_type'),
                n_jobs=self.n_jobs) for i, model_config in enumerate(self.model_config_extra_evaluators_dict)]
        return self._data_pool_extra_evaluators

    @property
    def data_val_extra_evaluators(self):
        if not hasattr(self, '_data_val_extra_evaluators'):
            self._data_val_extra_evaluators = [get_data(
                data_format=model_config['data_format'],
                path='%s/val.csv' % self.save_dir,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_extra_evaluators[i],
                features_combination=model_config.get('features_combination'),
                graph_kernel_type=model_config.get('graph_kernel_type'),
                n_jobs=self.n_jobs) for i, model_config in enumerate(self.model_config_extra_evaluators_dict)]
        return self._data_val_extra_evaluators

    @property
    def data_full_selector(self):
        if not hasattr(self, '_data_full_selector'):
            self._data_full_selector = get_data(
                data_format=self.model_config_selector_dict['data_format'],
                path=self.data_path,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_selector,
                features_combination=self.model_config_selector_dict.get('features_combination'),
                graph_kernel_type=self.model_config_selector_dict.get('graph_kernel_type'),
                n_jobs=self.n_jobs)
        return self._data_full_selector

    @property
    def data_full_evaluator(self):
        if self.yoked_learning:
            if not hasattr(self, '_data_full_evaluator'):
                self._data_full_evaluator = get_data(
                    data_format=self.model_config_evaluator_dict['data_format'],
                    path=self.data_path,
                    pure_columns=self.pure_columns,
                    mixture_columns=self.mixture_columns,
                    target_columns=self.target_columns,
                    feature_columns=self.feature_columns,
                    features_generator=self.features_generator_evaluator,
                    features_combination=self.model_config_evaluator_dict.get('features_combination'),
                    graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
                    n_jobs=self.n_jobs)
            return self._data_full_evaluator
        else:
            return self.data_full_selector

    @property
    def data_full_extra_evaluators(self) -> List:
        if not hasattr(self, '_data_full_extra_evaluators'):
            self._data_full_extra_evaluators = [get_data(
                data_format=model_config['data_format'],
                path=self.data_path,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_extra_evaluators[i],
                features_combination=model_config.get('features_combination'),
                graph_kernel_type=model_config.get('graph_kernel_type'),
                n_jobs=self.n_jobs) for i, model_config in enumerate(self.model_config_extra_evaluators_dict)]
        return self._data_full_extra_evaluators

    @property
    def features_generator_selector(self) -> Optional[List[FeaturesGenerator]]:
        fingerprints_class = self.model_config_selector_dict.get('fingerprints_class')
        radius = self.model_config_selector_dict.get('radius')
        num_bits = self.model_config_selector_dict.get('num_bits')
        if fingerprints_class is None:
            return None
        else:
            return [FeaturesGenerator(features_generator_name=fc,
                                      radius=radius,
                                      num_bits=num_bits) for fc in fingerprints_class]

    @property
    def features_generator_evaluator(self) -> Optional[List[FeaturesGenerator]]:
        if self.yoked_learning:
            fingerprints_class = self.model_config_evaluator_dict.get('fingerprints_class')
            radius = self.model_config_evaluator_dict.get('radius')
            num_bits = self.model_config_evaluator_dict.get('num_bits')
            if fingerprints_class is None:
                return None
            else:
                return [FeaturesGenerator(features_generator_name=fc,
                                          radius=radius,
                                          num_bits=num_bits) for fc in fingerprints_class]
        else:
            return self.features_generator_selector

    @property
    def features_generator_extra_evaluators(self) -> Optional[List[List[FeaturesGenerator]]]:
        results = []
        for model_config in self.model_config_extra_evaluators_dict:
            fingerprints_class = model_config.get('fingerprints_class')
            radius = model_config.get('radius')
            num_bits = model_config.get('num_bits')
            if fingerprints_class is None:
                results.append(None)
            else:
                results.append([FeaturesGenerator(features_generator_name=fc,
                                                  radius=radius,
                                                  num_bits=num_bits) for fc in fingerprints_class])
        return results

    @property
    def kernel_selector(self):
        return get_kernel(
            graph_kernel_type=self.model_config_selector_dict.get('graph_kernel_type'),
            mgk_files=self.model_config_selector_dict.get('mgk_files'),
            features_kernel_type=self.model_config_selector_dict.get('features_kernel_type'),
            features_hyperparameters=self.model_config_selector_dict.get('features_hyperparameters'),
            features_hyperparameters_file=self.model_config_selector_dict.get('features_hyperparameters_file'),
            dataset=self.data_full_selector,
            kernel_pkl_path='%s/kernel_selector.pkl' % self.save_dir,
        )

    @property
    def kernel_evaluator(self):
        if self.yoked_learning:
            return get_kernel(
                graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
                mgk_files=self.model_config_evaluator_dict.get('mgk_files'),
                features_kernel_type=self.model_config_evaluator_dict.get('features_kernel_type'),
                features_hyperparameters=self.model_config_evaluator_dict.get('features_hyperparameters'),
                features_hyperparameters_file=self.model_config_evaluator_dict.get('features_hyperparameters_file'),
                dataset=self.data_full_evaluator,
                kernel_pkl_path='%s/kernel_evaluator.pkl' % self.save_dir,
            )
        else:
            return self.kernel_selector

    @property
    def kernel_extra_evaluators(self) -> List:
        if not hasattr(self, '_kernel_extra_evaluators'):
            self._kernel_extra_evaluators = [get_kernel(
                graph_kernel_type=model_config.get('graph_kernel_type'),
                mgk_files=model_config.get('mgk_files'),
                features_kernel_type=model_config.get('features_kernel_type'),
                features_hyperparameters=model_config.get('features_hyperparameters'),
                features_hyperparameters_file=model_config.get('features_hyperparameters_file'),
                dataset=self.data_full_extra_evaluators[i],
                kernel_pkl_path='%s/kernel_extra_evaluator_%d.pkl' % (self.save_dir, i),
            ) for i, model_config in enumerate(self.model_config_extra_evaluators_dict)]
        return self._kernel_extra_evaluators

    def process_args(self) -> None:
        super().process_args()
        if self.stop_ratio is not None:
            if self.stop_size is None:
                self.stop_size = math.ceil(self.stop_ratio * (len(self.data_train_selector) + len(self.data_pool_selector)))
            else:
                self.stop_size = min(
                    self.stop_size,
                    math.ceil(self.stop_ratio * (len(self.data_train_selector) + len(self.data_pool_selector))))
            assert self.stop_size >= 2


class ActiveLearningContinueArgs(CommonArgs):
    stop_ratio: float = None
    """the ratio of molecules to stop the active learning."""
    stop_size: int = None
    """the number of molecules to stop the active learning."""


class ReEvaluateArgs(CommonArgs):
    model_config_evaluator: str
    """config file contain all information of the machine learning model for performance evaluation."""
    evaluator_id: int
    """the output id of the evaluator"""
    evaluate_stride: int = 100
    """evaluate model performance on the validation set when the size of the training set is an integer multiple of the 
    evaluation stride."""
    metrics: List[Metric]
    """the metrics to evaluate model performance."""

    data_public = None
    """Use public data sets."""
    data_path: str = None
    """the Path of input data CSV file."""
    pure_columns: List[str] = None
    """
    For pure compounds.
    Name of the columns containing single SMILES or InChI string.
    """
    mixture_columns: List[str] = None
    """
    For mixtures.
    Name of the columns containing multiple SMILES or InChI string and 
    corresponding concentration.
    example: ['C', 0.5, 'CC', 0.3]
    """
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    """
    feature_columns: List[str] = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    dataset_type: Literal['regression', 'classification', 'multiclass'] = None
    """
    Type of task.
    """
    full_val: bool = False
    """validate the performance of active learning on the full dataset."""

    @property
    def model_config_evaluator_dict(self) -> Dict:
        return json.loads(open(self.model_config_evaluator).read())

    @property
    def model_evaluator(self):
        if not hasattr(self, '_model_evaluator'):
            self._model_evaluator = get_model(
                data_format=self.model_config_evaluator_dict['data_format'],
                dataset_type=self.dataset_type,
                model=self.model_config_evaluator_dict.get('model'),
                save_dir='%s/extra_evaluator_%d' % (self.save_dir, self.evaluator_id),
                loss_function=self.model_config_evaluator_dict.get('loss_function'),
                num_tasks=len(self.target_columns),
                multiclass_num_classes=self.model_config_evaluator_dict.get('loss_function') or 3,
                features_generator=self.features_generator_evaluator,
                no_features_scaling=self.model_config_evaluator_dict.get('no_features_scaling') or False,
                features_only=self.model_config_evaluator_dict.get('features_only') or False,
                features_size=self.data_al_evaluator.features_size(),
                epochs=self.model_config_evaluator_dict.get('epochs') or 30,
                depth=self.model_config_evaluator_dict.get('depth') or 3,
                hidden_size=self.model_config_evaluator_dict.get('hidden_size') or 300,
                ffn_num_layers=self.model_config_evaluator_dict.get('ffn_num_layers') or 2,
                ffn_hidden_size=self.model_config_evaluator_dict.get('ffn_hidden_size'),
                dropout=self.model_config_evaluator_dict.get('dropout') or 0.0,
                batch_size=self.model_config_evaluator_dict.get('batch_size') or 50,
                ensemble_size=self.model_config_evaluator_dict.get('ensemble_size') or 1,
                number_of_molecules=self.model_config_evaluator_dict.get('number_of_molecules') or 1,
                mpn_shared=self.model_config_evaluator_dict.get('mpn_shared') or False,
                atom_messages=self.model_config_evaluator_dict.get('atom_messages') or False,
                undirected=self.model_config_evaluator_dict.get('undirected') or False,
                class_balance=self.model_config_evaluator_dict.get('class_balance') or False,
                checkpoint_dir=self.model_config_evaluator_dict.get('checkpoint_dir'),
                checkpoint_frzn=self.model_config_evaluator_dict.get('checkpoint_frzn'),
                frzn_ffn_layers=self.model_config_evaluator_dict.get('frzn_ffn_layers') or 0,
                freeze_first_only=self.model_config_evaluator_dict.get('freeze_first_only') or False,
                kernel=self.kernel_evaluator,
                uncertainty_type=self.model_config_evaluator_dict.get('uncertainty_type'),
                alpha=self.model_config_evaluator_dict.get('alpha'),
                n_jobs=self.n_jobs,
                seed=self.seed,
                logger=self.logger
            )
        return self._model_evaluator

    @property
    def data_al_evaluator(self):
        if not hasattr(self, '_data_al_evaluator'):
            self._data_al_evaluator = get_data(
                data_format=self.model_config_evaluator_dict['data_format'],
                path='%s/train_al.csv' % self.save_dir,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_evaluator,
                features_combination=self.model_config_evaluator_dict.get('features_combination'),
                graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
                n_jobs=self.n_jobs)
        return self._data_al_evaluator

    @property
    def data_val_evaluator(self):
        if not hasattr(self, '_data_val_evaluator'):
            self._data_val_evaluator = get_data(
                data_format=self.model_config_evaluator_dict['data_format'],
                path='%s/val.csv' % self.save_dir,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_evaluator,
                features_combination=self.model_config_evaluator_dict.get('features_combination'),
                graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
                n_jobs=self.n_jobs)
        return self._data_val_evaluator

    @property
    def data_full_evaluator(self):
        if not hasattr(self, '_data_full_evaluator'):
            self._data_full_evaluator = get_data(
                data_format=self.model_config_evaluator_dict['data_format'],
                path=self.data_path,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_evaluator,
                features_combination=self.model_config_evaluator_dict.get('features_combination'),
                graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
                n_jobs=self.n_jobs)
        return self._data_full_evaluator

    @property
    def features_generator_evaluator(self) -> Optional[List[FeaturesGenerator]]:
        fingerprints_class = self.model_config_evaluator_dict.get('fingerprints_class')
        radius = self.model_config_evaluator_dict.get('radius')
        num_bits = self.model_config_evaluator_dict.get('num_bits')
        if fingerprints_class is None:
            return None
        else:
            return [FeaturesGenerator(features_generator_name=fc,
                                      radius=radius,
                                      num_bits=num_bits) for fc in fingerprints_class]

    @property
    def kernel_evaluator(self):
        return get_kernel(
            graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
            mgk_files=self.model_config_evaluator_dict.get('mgk_files'),
            features_kernel_type=self.model_config_evaluator_dict.get('features_kernel_type'),
            features_hyperparameters=self.model_config_evaluator_dict.get('features_hyperparameters'),
            features_hyperparameters_file=self.model_config_evaluator_dict.get('features_hyperparameters_file'),
            dataset=self.data_full_evaluator,
            kernel_pkl_path='%s/kernel_extra_evaluator_%d.pkl' % (self.save_dir, self.evaluator_id),
        )

    def process_args(self) -> None:
        super().process_args()
        if self.data_public == 'freesolv':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['freesolv']
            self.dataset_type = 'regression'
        elif self.data_public == 'delaney':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['logSolubility']
            self.dataset_type = 'regression'
        elif self.data_public == 'lipo':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['lipo']
            self.dataset_type = 'regression'
        elif self.data_public == 'pdbbind_refined':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['-logKd/Ki']
            self.dataset_type = 'regression'
        elif self.data_public == 'pdbbind_full':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['-logKd/Ki']
            self.dataset_type = 'regression'
        elif self.data_public in ['ld50_zhu', 'caco2_wang', 'solubility_aqsoldb', 'ppbr_az', 'vdss_lombardo',
                                  'Half_Life_Obach', 'Clearance_Hepatocyte_AZ']:
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['Drug']
            self.target_columns = ['Y']
            self.dataset_type = 'regression'
        elif self.data_public == 'bbbp':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['p_np']
            self.dataset_type = 'classification'
        elif self.data_public == 'bace':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['mol']
            self.target_columns = ['Class']
            self.dataset_type = 'classification'
        elif self.data_public == 'hiv':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['HIV_active']
            self.dataset_type = 'classification'
        elif self.data_public in ['ames', 'carcinogens_lagunin', 'dili', 'herg', 'skin', 'hia_hou', 'pgp_broccatelli',
                                  'bioavailability_ma', 'clintox', 'bbb_martins', 'CYP1A2_Veith',
                                  'CYP2C9_Substrate_CarbonMangels', 'CYP2C9_Veith', 'CYP2C19_Veith',
                                  'CYP2D6_Substrate_CarbonMangels', 'CYP2D6_Veith', 'CYP3A4_Veith',
                                  'CYP3A4_Substrate_CarbonMangels']:
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['Drug']
            self.target_columns = ['Y']
            self.dataset_type = 'classification'

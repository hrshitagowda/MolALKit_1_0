#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

CWD = os.path.dirname(os.path.abspath(__file__))
from tap import Tap
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from logging import Logger
import json
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, DotProduct
from mgktools.features_mol import FeaturesGenerator
from alb.logging import create_logger
from alb.data.utils import split_data
from alb.utils import get_data, get_model, get_kernel

Metric = Literal['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc',
                 'rmse', 'mae', 'mse', 'r2', 'max']


class CommonArgs(Tap):
    save_dir: str
    """The output directory."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    quiet: bool = False
    """Whether the stream handler should be quiet (i.e., print only important info)."""
    logger_name: str = 'alb_output'
    """The prefix of the output logger file: verbose.log and quite.log"""
    seed: int = 0
    """random seed."""

    def process_args(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = create_logger(self.logger_name, save_dir=self.save_dir, quiet=self.quiet)


class DatasetArgs(CommonArgs):
    data_public: Literal['freesolv', 'delaney', 'clintox', 'bace', 'bbbp'] = None
    """Use public data sets."""
    data_path: str = None
    """The Path of input data CSV file."""
    data_path_training: str = None
    """The Path of input data CSV file for training set."""
    data_path_pool: str = None
    """The Path of input data CSV file for pool set."""
    data_path_val: str = None
    """The Path of input data CSV file for validation set."""
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
    normalize_fingerprints: bool = False
    """Nomralize the molecular features_mol."""
    normalize_features_add: bool = False
    """Nomralize the additonal features_mol."""
    split_type: Literal['random', 'scaffold_random', 'scaffold_order'] = 'random'
    """Method of splitting the data into active learning/validation."""
    split_sizes: List[float] = None
    """Split proportions for active learning/validation sets."""

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
        elif self.data_public == 'clintox':
            self.data_path = os.path.join(CWD, 'data', '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['FDA_APPROVED', 'CT_TOX']
            self.dataset_type = 'classification'
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

        if self.split_type == 'scaffold':
            assert len(self.pure_columns) == 1
            assert self.mixture_columns is None

        if self.dataset_type != 'regression':
            assert len(self.target_columns) == 1

        if self.data_path is not None:
            assert self.data_path_val is None and self.data_path_training is None and self.data_path_pool is None
            df = pd.read_csv(self.data_path)
            al_index, val_index = split_data(smiles=df[self.pure_columns[0]],
                                             # targets=df[self.target_columns[0]],
                                             split_type=self.split_type,
                                             sizes=self.split_sizes,
                                             seed=self.seed,
                                             logger=self.logger)
            df[df.index.isin(val_index)].to_csv('%s/val.csv' % self.save_dir, index=False)
            df_al = df[df.index.isin(al_index)]
            if self.dataset_type == 'regression':
                train_index, pool_index = split_data(smiles=df_al[self.pure_columns[0]],
                                                     split_type='random',
                                                     sizes=[2 / len(df_al), 1 - 2 / len(df_al)])
            else:
                train_index, pool_index = split_data(smiles=df_al[self.pure_columns[0]],
                                                     targets=df_al[self.target_columns[0]],
                                                     split_type='class',
                                                     n_samples_per_class=1)
            df_al.iloc[train_index].to_csv('%s/train_init.csv' % self.save_dir, index=False)
            df_al.iloc[pool_index].to_csv('%s/pool_init.csv' % self.save_dir, index=False)


class MarginalizedGraphKernelArgs(Tap):
    mgk_type: Literal['graph'] = None
    """The type of graph kernel to use."""
    mgk_hyperparameters_file: str = None
    """hyperparameters file for graph kernel."""


class FingerprintsKernelArgs(Tap):
    fingerprints_kernel_type: Literal['rbf', 'linear'] = None
    """The type of fingerprints kernel to use."""
    features_hyperparameters: List[float] = None
    """hyperparameters for fingperprints."""
    features_hyperparameters_file: str = None
    """JSON file contains features hyperparameters"""


class KernelArgs(CommonArgs, MarginalizedGraphKernelArgs, FingerprintsKernelArgs):
    pre_computed: bool = False
    """Whether use the pre-computed kernel."""

    @property
    def kernel_pkl(self) -> str:
        return os.path.join(self.save_dir, 'kernel.pkl')


class ModelArgs(Tap):
    model_config_selector: str
    """config file contain all information of the machine learning model."""
    model_config_evaluator: str = None
    """config file contain all information of the machine learning model for performance evaluation."""

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


class ActiveLearningArgs(DatasetArgs, KernelArgs, ModelArgs):
    save_dir: str
    """The output directory."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    data_path: str = None
    """The Path of input data CSV file."""
    learning_type: Literal['explorative', 'exploitive', 'EI', 'passive']
    """The learning type to be performed."""
    metrics: List[Metric]
    """The metrics to evaluate model performance."""
    evaluate_stride: int = 100
    """Evaluate model performance on the validation set after no. steps of active learning."""

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
                    n_jobs=self.n_jobs,
                    seed=self.seed,
                    logger=self.logger
                )
            return self._model_evaluator
        else:
            return self.model_selector

    @property
    def data_train_selector(self):
        if not hasattr(self, '_data_train_selector'):
            self._data_train_selector = get_data(data_format=self.model_config_selector_dict['data_format'],
                                                 path='%s/train_init.csv' % self.save_dir,
                                                 pure_columns=self.pure_columns,
                                                 mixture_columns=self.mixture_columns,
                                                 target_columns=self.target_columns,
                                                 feature_columns=self.feature_columns,
                                                 features_generator=self.features_generator_selector,
                                                 n_jobs=self.n_jobs)
        return self._data_train_selector

    @property
    def data_pool_selector(self):
        if not hasattr(self, '_data_pool_selector'):
            self._data_pool_selector = get_data(data_format=self.model_config_selector_dict['data_format'],
                                                path='%s/pool_init.csv' % self.save_dir,
                                                pure_columns=self.pure_columns,
                                                mixture_columns=self.mixture_columns,
                                                target_columns=self.target_columns,
                                                feature_columns=self.feature_columns,
                                                features_generator=self.features_generator_selector,
                                                n_jobs=self.n_jobs)
        return self._data_pool_selector

    @property
    def data_train_evaluator(self):
        if self.yoked_learning:
            if not hasattr(self, '_data_train_evaluator'):
                self._data_train_evaluator = get_data(data_format=self.model_config_evaluator_dict['data_format'],
                                                      path='%s/train_init.csv' % self.save_dir,
                                                      pure_columns=self.pure_columns,
                                                      mixture_columns=self.mixture_columns,
                                                      target_columns=self.target_columns,
                                                      feature_columns=self.feature_columns,
                                                      features_generator=self.features_generator_evaluator,
                                                      n_jobs=self.n_jobs)
            return self._data_train_evaluator
        else:
            return self.data_train_selector

    @property
    def data_pool_evaluator(self):
        if self.yoked_learning:
            if not hasattr(self, '_data_pool_evaluator'):
                self._data_pool_evaluator = get_data(data_format=self.model_config_evaluator_dict['data_format'],
                                                     path='%s/pool_init.csv' % self.save_dir,
                                                     pure_columns=self.pure_columns,
                                                     mixture_columns=self.mixture_columns,
                                                     target_columns=self.target_columns,
                                                     feature_columns=self.feature_columns,
                                                     features_generator=self.features_generator_evaluator,
                                                     n_jobs=self.n_jobs)
            return self._data_pool_evaluator
        else:
            return self.data_pool_selector

    @property
    def data_val_evaluator(self):
        if not hasattr(self, '_data_val_evaluator'):
            self._data_val_evaluator = get_data(data_format=self.model_config_evaluator_dict['data_format'],
                                                path='%s/val.csv' % self.save_dir,
                                                pure_columns=self.pure_columns,
                                                mixture_columns=self.mixture_columns,
                                                target_columns=self.target_columns,
                                                feature_columns=self.feature_columns,
                                                features_generator=self.features_generator_evaluator,
                                                n_jobs=self.n_jobs)
        return self._data_val_evaluator

    @property
    def data_full_selector(self):
        if not hasattr(self, '_data_full'):
            self._data_full_selector = get_data(data_format=self.model_config_selector_dict['data_format'],
                                                path=self.data_path,
                                                pure_columns=self.pure_columns,
                                                mixture_columns=self.mixture_columns,
                                                target_columns=self.target_columns,
                                                feature_columns=self.feature_columns,
                                                features_generator=self.features_generator_selector,
                                                n_jobs=self.n_jobs)
        return self._data_full_selector

    @property
    def data_full_evaluator(self):
        if self.yoked_learning:
            if not hasattr(self, '_data_full_evaluator'):
                self._data_full_evaluator = get_data(data_format=self.model_config_evaluator_dict['data_format'],
                                                     path=self.data_path,
                                                     pure_columns=self.pure_columns,
                                                     mixture_columns=self.mixture_columns,
                                                     target_columns=self.target_columns,
                                                     feature_columns=self.feature_columns,
                                                     features_generator=self.features_generator_evaluator,
                                                     n_jobs=self.n_jobs)
            return self._data_full_evaluator
        else:
            return self.data_full_selector

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
    def kernel_selector(self):
        return get_kernel(
            graph_kernel_type=self.model_config_selector_dict.get('graph_kernel_type'),
            mgk_files=self.model_config_selector_dict.get('mgk_files'),
            features_kernel_type=self.model_config_selector_dict.get('features_kernel_type'),
            rbf_length_scale=self.model_config_selector_dict.get('rbf_length_scale'),
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
                rbf_length_scale=self.model_config_evaluator_dict.get('rbf_length_scale'),
                features_hyperparameters_file=self.model_config_evaluator_dict.get('features_hyperparameters_file'),
                dataset=self.data_full_evaluator,
                kernel_pkl_path='%s/kernel_evaluator.pkl' % self.save_dir,
            )
        else:
            return self.kernel_selector

    def process_args(self) -> None:
        super().process_args()

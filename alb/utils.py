#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

CWD = os.path.dirname(os.path.abspath(__file__))
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple, Callable
from logging import Logger
import pickle
from sklearn.gaussian_process.kernels import RBF, DotProduct
from mgktools.kernels.utils import get_kernel_config
from mgktools.data.data import Dataset


class EmptyLogger:
    def debug(self, info):
        return

    def info(self, info):
        return


def get_data(data_format: Literal['mgktools', 'chemprop', 'fingerprints'],
             # dataset_type: Literal['regression', 'classification', 'multiclass'],
             # model: Literal['random_forest', 'gaussian_process', 'support_vector_machine'],
             path: str,
             pure_columns: List[str] = None,
             mixture_columns: List[str] = None,
             target_columns: List[str] = None,
             feature_columns: List[str] = None,
             features_generator: List[Union[str, Callable]] = None,
             features_combination: Literal['concat', 'mean'] = None,
             graph_kernel_type: Literal['graph', 'pre-computed'] = None,
             n_jobs: int = 8):
    if data_format == 'fingerprints':
        from alb.data.utils import get_data
        return get_data(path=path,
                        pure_columns=pure_columns,
                        mixture_columns=mixture_columns,
                        target_columns=target_columns,
                        feature_columns=feature_columns,
                        features_generator=features_generator,
                        features_combination=features_combination,
                        n_jobs=n_jobs)
    elif data_format == 'chemprop':
        from chemprop.data.utils import get_data
        assert mixture_columns is None
        assert feature_columns is None
        return get_data(path=path,
                        smiles_columns=pure_columns,
                        target_columns=target_columns,
                        features_generator=features_generator)
    elif data_format == 'mgktools':
        assert graph_kernel_type is not None
        from mgktools.data.data import get_data
        return get_data(path=path,
                        pure_columns=pure_columns,
                        mixture_columns=mixture_columns,
                        target_columns=target_columns,
                        feature_columns=feature_columns,
                        features_generator=features_generator,
                        features_combination=features_combination,
                        mixture_type='single_graph',
                        graph_kernel_type=graph_kernel_type,
                        n_jobs=n_jobs)
    else:
        raise ValueError('input error')


def get_model(data_format: Literal['mgktools', 'chemprop', 'fingerprints'],
              dataset_type: Literal['regression', 'classification', 'multiclass'],
              model: Literal['random_forest', 'gaussian_process', 'support_vector_machine'],
              save_dir: str = None,
              loss_function: Literal['mse', 'bounded_mse', 'binary_cross_entropy', 'cross_entropy', 'mcc', 'sid',
                                     'wasserstein', 'mve', 'evidential', 'dirichlet'] = None,
              num_tasks: int = 1,
              multiclass_num_classes: int = 3,
              features_generator=None,
              no_features_scaling: bool = False,
              features_only: bool = False,
              features_size: int = 0,
              epochs: int = 30,
              depth: int = 3,
              hidden_size: int = 300,
              ffn_num_layers: int = 2,
              ffn_hidden_size: int = None,
              dropout: float = 0.0,
              batch_size: int = 50,
              ensemble_size: int = 1,
              number_of_molecules: int = 1,
              mpn_shared: bool = False,
              atom_messages: bool = False,
              undirected: bool = False,
              class_balance: bool = False,
              checkpoint_dir: str = None,
              checkpoint_frzn: str = None,
              frzn_ffn_layers: int = 0,
              freeze_first_only: bool = False,
              kernel=None,
              uncertainty_type: Literal['value', 'uncertainty'] = None,
              alpha: Union[float, str] = 1e-8,
              n_jobs: int = 8,
              seed: int = 0,
              logger: Logger = None,
              ):
    if alpha.__class__ == str:
        alpha = float(open(alpha).read())

    if data_format == 'fingerprints':
        if model == 'random_forest':
            if dataset_type == 'regression':
                from alb.models.random_forest.RandomForestRegressor import RFRegressor
                return RFRegressor(random_state=seed)
            else:
                from alb.models.random_forest.RandomForestClassifier import RFClassifier
                return RFClassifier(random_state=seed)
        elif model == 'logistic_regression':
            assert dataset_type == 'classification'
            from alb.models.logistic_regression.LogisticRegression import LogisticRegressor
            return LogisticRegressor(random_state=seed)
        elif model == 'gaussian_process_regression':
            assert dataset_type in ['regression', 'classification']
            assert uncertainty_type is not None
            from alb.models.gaussian_process.GaussianProcessRegressor import GPRegressor
            return GPRegressor(kernel=kernel, alpha=alpha, optimizer=None, uncertainty_type=uncertainty_type)
        elif model == 'gaussian_process_classification':
            assert dataset_type == 'classification'
            from alb.models.gaussian_process.GaussianProcessClassifier import GPClassifier
            return GPClassifier(kernel=kernel, optimizer=None)
        elif model == 'support_vector_machine':
            assert dataset_type == 'classification'
            from alb.models.support_vector.SupportVectorClassifier import SVClassifier
            return SVClassifier(kernel=kernel, probability=True)
        else:
            raise ValueError(f'unknown model: {model}')
    elif data_format == 'chemprop':
        from alb.models.mpnn.mpnn import MPNN
        return MPNN(save_dir=save_dir,
                    dataset_type=dataset_type,
                    loss_function=loss_function,
                    num_tasks=num_tasks,
                    multiclass_num_classes=multiclass_num_classes,
                    features_generator=features_generator,
                    no_features_scaling=no_features_scaling,
                    features_only=features_only,
                    features_size=features_size,
                    epochs=epochs,
                    depth=depth,
                    hidden_size=hidden_size,
                    ffn_num_layers=ffn_num_layers,
                    ffn_hidden_size=ffn_hidden_size,
                    dropout=dropout,
                    batch_size=batch_size,
                    ensemble_size=ensemble_size,
                    number_of_molecules=number_of_molecules,
                    mpn_shared=mpn_shared,
                    atom_messages=atom_messages,
                    undirected=undirected,
                    class_balance=class_balance,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_frzn=checkpoint_frzn,
                    frzn_ffn_layers=frzn_ffn_layers,
                    freeze_first_only=freeze_first_only,
                    n_jobs=n_jobs,
                    seed=seed,
                    logger=EmptyLogger())
    elif data_format == 'mgktools':
        if model == 'gaussian_process_regression':
            assert dataset_type in ['regression', 'classification']
            assert uncertainty_type is not None
            from alb.models.gaussian_process.GaussianProcessRegressor import GPRegressor
            return GPRegressor(kernel=kernel, alpha=alpha, optimizer=None, uncertainty_type=uncertainty_type)
        elif model == 'gaussian_process_classification':
            assert dataset_type == 'classification'
            from alb.models.gaussian_process.GaussianProcessClassifier import GPClassifier
            return GPClassifier(kernel=kernel, optimizer=None)
        elif model == 'support_vector_machine':
            assert dataset_type == 'classification'
            from alb.models.support_vector.SupportVectorClassifier import SVClassifier
            return SVClassifier(kernel=kernel, probability=True)
        else:
            raise ValueError(f'unknown model: {model}')
    else:
        raise ValueError(f'unknown data_format {data_format}')


def get_kernel(graph_kernel_type: Literal['graph', 'pre-computed'] = None,
               mgk_files: List[str] = None,
               features_kernel_type: Literal['linear', 'dot_product', 'rbf'] = None,
               features_hyperparameters: Union[float, List[float]] = None,
               features_hyperparameters_file: str = None,
               dataset: Dataset = None,
               kernel_pkl_path: str = None,
               ):
    if mgk_files is None:
        assert graph_kernel_type is None
        # no graph kernel involved.
        if features_kernel_type is None:
            return None
        elif features_kernel_type == 'linear':
            return 'linear'
        elif features_kernel_type == 'dot_product':
            if features_hyperparameters.__class__ == list:
                assert len(features_hyperparameters) == 1
                sigma_0 = features_hyperparameters[0]
            else:
                sigma_0 = features_hyperparameters
            return DotProduct(sigma_0=sigma_0)
        elif features_kernel_type == 'rbf':
            return RBF(length_scale=features_hyperparameters)
        else:
            raise ValueError
    else:
        if graph_kernel_type == 'graph':
            return get_kernel_config(
                dataset=dataset,
                graph_kernel_type='graph',
                mgk_hyperparameters_files=mgk_files,
                features_kernel_type=features_kernel_type,
                features_hyperparameters=features_hyperparameters,
                features_hyperparameters_bounds="fixed",
                features_hyperparameters_file=features_hyperparameters_file
            ).kernel
        elif graph_kernel_type == 'pre-computed':
            assert kernel_pkl_path is not None
            if os.path.exists(kernel_pkl_path):
                return get_kernel_config(
                    dataset=dataset,
                    graph_kernel_type='pre-computed',
                    features_kernel_type=features_kernel_type,
                    features_hyperparameters=features_hyperparameters,
                    features_hyperparameters_bounds="fixed",
                    features_hyperparameters_file=features_hyperparameters_file,
                    kernel_pkl=kernel_pkl_path
                ).kernel
            else:
                dataset.graph_kernel_type = 'graph'
                kernel_config = get_kernel_config(
                    dataset=dataset,
                    graph_kernel_type='graph',
                    mgk_hyperparameters_files=mgk_files,
                    features_kernel_type=features_kernel_type,
                    features_hyperparameters=features_hyperparameters,
                    features_hyperparameters_bounds="fixed",
                    features_hyperparameters_file=features_hyperparameters_file
                )
                kernel_dict = kernel_config.get_kernel_dict(dataset.X, dataset.X_repr.ravel())
                dataset.graph_kernel_type = 'pre-computed'
                pickle.dump(kernel_dict, open(kernel_pkl_path, 'wb'), protocol=4)
                return get_kernel_config(
                    dataset=dataset,
                    graph_kernel_type='pre-computed',
                    features_kernel_type=features_kernel_type,
                    features_hyperparameters=features_hyperparameters,
                    features_hyperparameters_bounds="fixed",
                    features_hyperparameters_file=features_hyperparameters_file,
                    kernel_dict=kernel_dict
                ).kernel
        else:
            raise ValueError

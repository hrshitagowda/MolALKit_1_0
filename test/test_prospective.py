#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
from molalkit.args import ActiveLearningArgs
from molalkit.al.learner import ActiveLearner
from molalkit.data.datasets import DATA_DIR


CWD = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize('batch_size', ['1', '5', '20'])
def test_classification_no_val(batch_size):
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_path_training', f'{DATA_DIR}/carcinogens_lagunin.csv',
        '--data_path_pool', f'{DATA_DIR}/skin.csv',
        '--pure_columns', 'Drug',
        '--target_columns', 'Y',
        '--learning_type', 'explorative',
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--split_type', 'random',
        '--split_sizes', '1.0', '0.0',
        '--batch_size', batch_size,
        '--stop_size', '300',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = ActiveLearner(save_dir=args.save_dir,
                                   selection_method=args.selection_method,
                                   forgetter=args.forgetter,
                                   model_selector=args.model_selector,
                                   dataset_train_selector=args.data_train_selector,
                                   dataset_pool_selector=args.data_pool_selector,
                                   dataset_val_selector=args.data_val_selector,
                                   metrics=args.metrics,
                                   top_k_id=args.top_k_id,
                                   model_evaluators=args.model_evaluators,
                                   dataset_train_evaluators=args.data_train_evaluators,
                                   dataset_pool_evaluators=args.data_pool_evaluators,
                                   dataset_val_evaluators=args.data_val_evaluators,
                                   yoked_learning_only=args.yoked_learning_only,
                                   stop_size=args.stop_size,
                                   evaluate_stride=args.evaluate_stride,
                                   kernel=args.kernel_selector,
                                   save_cpt_stride=args.save_cpt_stride,
                                   seed=args.seed,
                                   logger=args.logger)
    alr = active_learner.run_prospective()
    batch_size = int(batch_size)
    assert len(alr.id_add) == batch_size
    ids = [data.id for data in active_learner.dataset_train_selector[-batch_size:]]
    assert set(ids) == set(alr.id_add)


@pytest.mark.parametrize('batch_size', ['1', '5', '20'])
def test_classification_no_pool(batch_size):
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_path_training', f'{DATA_DIR}/carcinogens_lagunin.csv',
        '--data_path_val', f'{DATA_DIR}/skin.csv',
        '--pure_columns', 'Drug',
        '--target_columns', 'Y',
        '--learning_type', 'explorative',
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--split_type', 'random',
        '--split_sizes', '1.0', '0.0',
        '--batch_size', batch_size,
        '--stop_size', '300',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = ActiveLearner(save_dir=args.save_dir,
                                   selection_method=args.selection_method,
                                   forgetter=args.forgetter,
                                   model_selector=args.model_selector,
                                   dataset_train_selector=args.data_train_selector,
                                   dataset_pool_selector=args.data_pool_selector,
                                   dataset_val_selector=args.data_val_selector,
                                   metrics=args.metrics,
                                   top_k_id=args.top_k_id,
                                   model_evaluators=args.model_evaluators,
                                   dataset_train_evaluators=args.data_train_evaluators,
                                   dataset_pool_evaluators=args.data_pool_evaluators,
                                   dataset_val_evaluators=args.data_val_evaluators,
                                   yoked_learning_only=args.yoked_learning_only,
                                   stop_size=args.stop_size,
                                   evaluate_stride=args.evaluate_stride,
                                   kernel=args.kernel_selector,
                                   save_cpt_stride=args.save_cpt_stride,
                                   seed=args.seed,
                                   logger=args.logger)
    alr = active_learner.run_prospective()
    batch_size = int(batch_size)
    assert len(alr.id_add) == batch_size
    ids = [data.id for data in active_learner.dataset_train_selector[-batch_size:]]
    assert set(ids) == set(alr.id_add)

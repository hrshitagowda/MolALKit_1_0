#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
from molalkit.args import ActiveLearningArgs
from model.test_model import run, al_results_check


CWD = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize('split', ['random', 'scaffold_order', 'scaffold_random'])
def test_classification(split):
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--metrics', 'roc-auc',
        '--learning_type', 'explorative',
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--split_type', split,
        '--split_sizes', '0.5', '0.5',
        '--evaluate_stride', '1',
        '--stop_size', '5',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert len(active_learner.active_learning_traj.results) == 3
    al_results_check(save_dir)


def test_classification_full():
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--metrics', 'roc-auc',
        '--learning_type', 'explorative',
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--full_val',
        '--evaluate_stride', '1',
        '--stop_size', '5',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert (len(active_learner.dataset_train_selector) + len(active_learner.dataset_pool_selector) ==
            len(active_learner.dataset_val_selector))
    assert len(active_learner.active_learning_traj.results) == 3
    al_results_check(save_dir)


def test_classification_no_val():
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--learning_type', 'explorative',
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--split_type', 'random',
        '--split_sizes', '1.0', '0.0',
        '--stop_size', '5',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    assert args.data_val_selector is None
    active_learner = run(args)
    assert active_learner.dataset_val_selector is None
    assert len(active_learner.active_learning_traj.results) == 3
    al_results_check(save_dir)


@pytest.mark.parametrize('split', ['random', 'scaffold_order', 'scaffold_random'])
def test_regression(split):
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'test_regression',
        '--metrics', 'rmse',
        '--learning_type', 'explorative',
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--split_type', split,
        '--split_sizes', '0.5', '0.5',
        '--evaluate_stride', '1',
        '--stop_size', '5',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert len(active_learner.active_learning_traj.results) == 3
    al_results_check(save_dir)


def test_regression_full():
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'test_regression',
        '--metrics', 'rmse',
        '--learning_type', 'explorative',
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--full_val',
        '--evaluate_stride', '1',
        '--stop_size', '5',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert (len(active_learner.dataset_train_selector) + len(active_learner.dataset_pool_selector) ==
            len(active_learner.dataset_val_selector))
    assert len(active_learner.active_learning_traj.results) == 3
    al_results_check(save_dir)


def test_regression_no_val():
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'test_regression',
        '--learning_type', 'explorative',
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--split_type', 'random',
        '--split_sizes', '1.0', '0.0',
        '--stop_size', '5',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    assert args.data_val_selector is None
    active_learner = run(args)
    assert active_learner.dataset_val_selector is None
    assert len(active_learner.active_learning_traj.results) == 3
    al_results_check(save_dir)

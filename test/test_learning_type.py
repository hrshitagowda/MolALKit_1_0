#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
from molalkit.args import ActiveLearningArgs
from model.test_model import run, al_results_check


CWD = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize('learning_type', ['explorative', 'passive'])
def test_classification(learning_type):
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--metrics', 'roc-auc',
        '--learning_type', learning_type,
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--split_type', 'scaffold_order',
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


@pytest.mark.parametrize('learning_type', ['explorative', 'passive'])
def test_regression(learning_type):
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'test_regression',
        '--metrics', 'rmse',
        '--learning_type', learning_type,
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--split_type', 'scaffold_order',
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


@pytest.mark.parametrize('learning_type', ['exploitive'])
@pytest.mark.parametrize('exploitive_target', ['min', 'max', '1.0'])
def test_regression_exploitive(learning_type, exploitive_target):
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'test_regression',
        '--metrics', 'rmse',
        '--learning_type', learning_type,
        '--exploitive_target', exploitive_target,
        '--top_k', '0.1',
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--split_type', 'scaffold_order',
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

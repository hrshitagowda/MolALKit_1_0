#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
from molalkit.args import ActiveLearningArgs
from model.test_model import run, al_results_check

CWD = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize('params_set', [('passive', 'naive', None),
                                        ('passive', 'cluster', None),
                                        ('explorative', 'naive', None),
                                        ('explorative', 'cluster', None),
                                        ('explorative', 'naive', '50'),
                                        ('explorative', 'cluster', '50')])
def test_classification(params_set):
    learning_type, batch_mode, n_query = params_set
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--metrics', 'roc-auc',
        '--learning_type', learning_type,
        '--batch_mode', batch_mode,
        '--batch_size', '5',
        '--model_config_selector', 'GaussianProcessRegressionDecisionBoundaryUncertainty_RBFKernelRDKitNorm_Config',
        '--split_type', 'scaffold_order',
        '--split_sizes', '0.5', '0.5',
        '--evaluate_stride', '1',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    if n_query is not None:
        arguments.extend(['--n_query', n_query])
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert len(active_learner.active_learning_traj.results) == 28
    al_results_check(save_dir)


@pytest.mark.parametrize('params_set', [('passive', 'naive', None),
                                        ('passive', 'cluster', None),
                                        ('explorative', 'naive', None),
                                        ('explorative', 'cluster', None),
                                        ('exploitive', 'naive', 'min'),
                                        ('exploitive', 'naive', 'max'),
                                        ('exploitive', 'naive', '1.0'),
                                        ('exploitive', 'cluster', 'min'),
                                        ('exploitive', 'cluster', 'max'),
                                        ('exploitive', 'cluster', '1.0')])
def test_regression1(params_set):
    learning_type, batch_mode, exploitive_target = params_set
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'test_regression',
        '--metrics', 'rmse',
        '--learning_type', learning_type,
        '--batch_mode', batch_mode,
        '--batch_size', '5',
        '--model_config_selector', 'GaussianProcessRegressionDecisionBoundaryUncertainty_RBFKernelRDKitNorm_Config',
        '--split_type', 'scaffold_order',
        '--split_sizes', '0.5', '0.5',
        '--evaluate_stride', '1',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    if exploitive_target is not None:
        arguments.extend(['--exploitive_target', exploitive_target, '--top_k', '0.1'])
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert len(active_learner.active_learning_traj.results) == 21
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
from molalkit.args import ActiveLearningArgs
from model.test_model import run, al_results_check


CWD = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize('params', [('stop_size', '5', 3),
                                    ('stop_ratio', '0.1', 12),
                                    ('stop_cutoff', '0.5', 60)])
def test_classification(params):
    param1, param2, params3 = params
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--metrics', 'roc-auc',
        '--learning_type', 'explorative',
        '--model_config_selector', 'GaussianProcessRegressionDecisionBoundaryUncertainty_RBFKernelRDKitNorm_Config',
        '--split_type', 'scaffold_order',
        '--split_sizes', '0.5', '0.5',
        f'--{param1}', param2,
        '--evaluate_stride', '1',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert len(active_learner.active_learning_traj.results) == params3
    al_results_check(save_dir)


@pytest.mark.parametrize('params', [('min', -5, 34),
                                    ('max', 0, 17),
                                    ('-5', 2, 44)])
def test_regression(params):
    params1, params2, params3 = params
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'test_regression',
        '--metrics', 'rmse',
        '--learning_type', 'exploitive',
        '--exploitive_target', params1,
        '--model_config_selector', 'GaussianProcessRegressionPosteriorUncertainty_RBFKernelRDKitNorm_Config',
        '--split_type', 'random',
        '--top_k', '0.1',
        '--split_sizes', '0.8', '0.2',
        '--init_size', '50',
        '--stop_cutoff', str(params2),
        '--evaluate_stride', '1',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert len(active_learner.active_learning_traj.results) == params3
    al_results_check(save_dir)

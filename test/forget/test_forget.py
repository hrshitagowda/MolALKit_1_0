#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
import sys
from molalkit.args import ActiveLearningArgs
from molalkit.al.run import run

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from model.test_model import al_results_check


@pytest.mark.parametrize('forget_protocol', ['forget_first', 'forget_random',
                                             'min_oob_uncertainty', 'max_oob_uncertainty'])
@pytest.mark.parametrize('forget_para_set', [('forget_ratio', '0.1'),
                                             ('forget_size', '10')])
def test_classification_rf(forget_protocol, forget_para_set):
    forget_para1, forget_para2 = forget_para_set
    save_dir = os.path.join(CWD, '..', 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--metrics', 'roc-auc',
        '--learning_type', 'explorative',
        '--forget_protocol', forget_protocol,
        f'--{forget_para1}', forget_para2,
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--split_type', 'scaffold_order',
        '--split_sizes', '0.5', '0.5',
        '--evaluate_stride', '1',
        '--stop_size', '200',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert len(active_learner.active_learning_traj.results) == 139
    al_results_check(save_dir)


@pytest.mark.parametrize('forget_protocol', ['min_oob_error'])
@pytest.mark.parametrize('forget_para_set', [('forget_ratio', '0.1'),
                                             ('forget_size', '10'),
                                             ('forget_cutoff', '0.1')])
def test_classification_rf_oob_error(forget_protocol, forget_para_set):
    forget_para1, forget_para2 = forget_para_set
    save_dir = os.path.join(CWD, '..', 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--metrics', 'roc-auc',
        '--learning_type', 'explorative',
        '--forget_protocol', forget_protocol,
        f'--{forget_para1}', forget_para2,
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--split_type', 'scaffold_order',
        '--split_sizes', '0.5', '0.5',
        '--evaluate_stride', '1',
        '--stop_size', '200',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert len(active_learner.active_learning_traj.results) == 139
    al_results_check(save_dir)


@pytest.mark.parametrize('model', ['GaussianProcessRegressionPosteriorUncertainty_RBFKernelRDKitNorm_Config',
                                   'GaussianProcessRegressionDecisionBoundaryUncertainty_RBFKernelRDKitNorm_Config'])
@pytest.mark.parametrize('forget_protocol', ['min_loo_error'])
@pytest.mark.parametrize('forget_para_set', [('forget_ratio', '0.1'),
                                             ('forget_size', '10'),
                                             ('forget_cutoff', '0.1')])
def test_classification_gpr(model, forget_protocol, forget_para_set):
    forget_para1, forget_para2 = forget_para_set
    save_dir = os.path.join(CWD, '..', 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--metrics', 'roc-auc',
        '--learning_type', 'explorative',
        '--forget_protocol', forget_protocol,
        f'--{forget_para1}', forget_para2,
        '--model_config_selector', model,
        '--split_type', 'scaffold_order',
        '--split_sizes', '0.5', '0.5',
        '--evaluate_stride', '1',
        '--stop_size', '200',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert len(active_learner.active_learning_traj.results) == 139
    al_results_check(save_dir)

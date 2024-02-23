#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
from molalkit.args import ActiveLearningArgs
from molalkit.models.configs import AVAILABLE_MODELS
from molalkit.al.run import run
from test_model import al_results_check


CWD = os.path.dirname(os.path.abspath(__file__))
MGK_CLASSIFICATION_MODELS = ['GPR-MGK-RDKit_Config']
for model in AVAILABLE_MODELS:
    if model.startswith('GaussianProcess') or model.startswith('SupportVectorMachine'):
        if 'MarginalizedGraphKernel' in model:
            MGK_CLASSIFICATION_MODELS.append(model)
MGK_REGRESSION_MODELS = ['GPR-MGK-RDKit_Config']
for model in AVAILABLE_MODELS:
    if model.startswith('GaussianProcessRegressionPosteriorUncertainty'):
        if 'MarginalizedGraphKernel' in model:
            MGK_REGRESSION_MODELS.append(model)


@pytest.mark.parametrize('model', MGK_CLASSIFICATION_MODELS)
def test_classification(model):
    save_dir = os.path.join(CWD, '..', 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--metrics', 'roc-auc',
        '--learning_type', 'explorative',
        '--model_config_selector', model,
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
    assert len(active_learner.active_learning_traj.results) == 4
    al_results_check(save_dir)


@pytest.mark.parametrize('model', MGK_CLASSIFICATION_MODELS)
def test_classification_cv(model):
    save_dir = os.path.join(CWD, '..', 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--metrics', 'roc-auc',
        '--learning_type', 'explorative',
        '--model_config_selector', model,
        '--split_type', 'scaffold_order',
        '--split_sizes', '0.5', '0.5',
        '--evaluate_stride', '1',
        '--init_size', '10000',
        '--output_details',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert len(active_learner.active_learning_traj.results) == 1
    al_results_check(save_dir)


@pytest.mark.parametrize('model', MGK_REGRESSION_MODELS)
def test_regression(model):
    save_dir = os.path.join(CWD, '..', 'test')
    arguments = [
        '--data_public', 'test_regression',
        '--metrics', 'rmse',
        '--learning_type', 'explorative',
        '--model_config_selector', model,
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
    assert len(active_learner.active_learning_traj.results) == 4
    al_results_check(save_dir)


@pytest.mark.parametrize('model', MGK_REGRESSION_MODELS)
def test_regression_cv(model):
    save_dir = os.path.join(CWD, '..', 'test')
    arguments = [
        '--data_public', 'test_regression',
        '--metrics', 'rmse',
        '--learning_type', 'explorative',
        '--model_config_selector', model,
        '--split_type', 'scaffold_order',
        '--split_sizes', '0.5', '0.5',
        '--evaluate_stride', '1',
        '--init_size', '10000',
        '--output_details',
        '--seed', '0',
        '--save_dir', save_dir,
        '--n_jobs', '4'
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert len(active_learner.active_learning_traj.results) == 1
    al_results_check(save_dir)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
from molalkit.args import ActiveLearningArgs
from molalkit.models.configs import AVAILABLE_MODELS
from molalkit.al.run import molalkit_run, molalkit_run_from_cpt
from test_model import al_results_check


CWD = os.path.dirname(os.path.abspath(__file__))
DEEP_CLASSIFICATION_MODELS = []
for model in AVAILABLE_MODELS:
    if model.startswith('MLP') or model.startswith('DMPNN'):
        if 'BinaryClassification' in model:
            DEEP_CLASSIFICATION_MODELS.append(model)
DEEP_REGRESSION_MODELS = []
for model in AVAILABLE_MODELS:
    if model.startswith('MLP') or model.startswith('DMPNN'):
        if 'Regression_Evidential' in model or 'Regression_MVE' in model:
            DEEP_REGRESSION_MODELS.append(model)


@pytest.mark.parametrize('model', DEEP_CLASSIFICATION_MODELS)
def test_classification(model):
    save_dir = os.path.join(CWD, '..', 'test')
    assert not os.path.exists('%s/al.pkl' % save_dir)
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
        '--n_jobs', '4',
        '--save_cpt_stride', '1'
    ]
    active_learner = molalkit_run(arguments)
    assert os.path.exists('%s/al.pkl' % save_dir)
    assert len(active_learner.active_learning_traj.results) == 4
    al_results_check(save_dir)
    arguments = [
        '--save_dir', save_dir,
        '--stop_size', '6',
        '--max_iter', '10'
    ]
    molalkit_run_from_cpt(arguments)
    os.unlink('%s/al.pkl' % save_dir)
    

@pytest.mark.parametrize('model', DEEP_CLASSIFICATION_MODELS)
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
    active_learner = molalkit_run(arguments)
    assert len(active_learner.active_learning_traj.results) == 1
    al_results_check(save_dir)


@pytest.mark.parametrize('model', DEEP_REGRESSION_MODELS)
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
    active_learner = molalkit_run(arguments)
    assert len(active_learner.active_learning_traj.results) == 4
    al_results_check(save_dir)


@pytest.mark.parametrize('model', DEEP_REGRESSION_MODELS)
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
    active_learner = molalkit_run(arguments)
    assert len(active_learner.active_learning_traj.results) == 1
    al_results_check(save_dir)

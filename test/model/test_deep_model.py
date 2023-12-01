#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
from molalkit.args import ActiveLearningArgs
from molalkit.models.configs import AVAILABLE_MODELS
from molalkit.al.run import run
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
    assert len(active_learner.active_learning_traj.results) == 3
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
    args = ActiveLearningArgs().parse_args(arguments)
    active_learner = run(args)
    assert len(active_learner.active_learning_traj.results) == 3
    al_results_check(save_dir)

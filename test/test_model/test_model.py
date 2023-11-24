#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
import shutil
import json
import pandas as pd
from molalkit.args import ActiveLearningArgs
from molalkit.models.configs import AVAILABLE_MODELS
from molalkit.al.run import run


def al_results_check(save_dir):
    df = pd.read_csv(f'{save_dir}/al_traj.csv')
    for i, row in df.iterrows():
        if i == 0:
            id_prior_al = set(json.loads(row['id_prior_al']))
            id_forgotten = set(json.loads(row['id_forgotten']))
            id_add = set(json.loads(row['id_add']))
        else:
            id_prior_al.update(id_add)
            id_prior_al.difference_update(id_forgotten)
            assert id_prior_al == set(json.loads(row['id_prior_al']))
            id_forgotten = set(json.loads(row['id_forgotten']))
            id_add = set(json.loads(row['id_add']))


CWD = os.path.dirname(os.path.abspath(__file__))
AVAILABLE_CLASSIFICATION_MODELS = []
for model in AVAILABLE_MODELS:
    if '2Mol' in model or 'MLP' in model or 'DMPNN' in model or 'MarginalizedGraph' in model:
        continue
    AVAILABLE_CLASSIFICATION_MODELS.append(model)
AVAILABLE_REGRESSION_MODELS = []
for model in AVAILABLE_MODELS:
    if '2Mol' in model or 'MLP' in model or 'DMPNN' in model or 'MarginalizedGraph' in model:
        continue
    if 'RandomForest' in model or 'GaussianProcessRegressionPosteriorUncertainty' in model:
        AVAILABLE_REGRESSION_MODELS.append(model)


@pytest.mark.parametrize('model', AVAILABLE_CLASSIFICATION_MODELS)
def test_classification(model):
    save_dir = os.path.join(CWD, 'test')
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
    shutil.rmtree(f'{save_dir}')


@pytest.mark.parametrize('model', AVAILABLE_REGRESSION_MODELS)
def test_regression(model):
    save_dir = os.path.join(CWD, 'test')
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
    shutil.rmtree(f'{save_dir}')

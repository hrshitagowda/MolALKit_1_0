#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
from molalkit.args import ActiveLearningArgs
from model.test_model import run, al_results_check


CWD = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize('teacher_model', ['RandomForest_RDKitNorm_Config'])
@pytest.mark.parametrize('student_model', ['DMPNN+RDKitNorm_BinaryClassification_Config'])
def test_classification(teacher_model, student_model):
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--metrics', 'roc-auc',
        '--learning_type', 'explorative',
        '--model_config_selector', teacher_model,
        '--model_config_evaluators', student_model,
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


@pytest.mark.parametrize('teacher_model', ['RandomForest_RDKitNorm_Config'])
@pytest.mark.parametrize('student_model', ['DMPNN+RDKitNorm_Regression_Config'])
def test_regression(teacher_model, student_model):
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'test_regression',
        '--metrics', 'rmse',
        '--learning_type', 'explorative',
        '--model_config_selector', teacher_model,
        '--model_config_evaluators', student_model,
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

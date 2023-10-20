# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append('%s/..' % CWD)
from molalkit.args import ActiveLearningArgs, ActiveLearningContinueArgs


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv'])
])
@pytest.mark.parametrize('modelset', [
    ('rf-morgan', 'RandomForest_Morgan_Config'),
    ('gpr-dpk-morgan', 'GaussianProcessRegressionUncertainty_DotProductKernel_Morgan_Config'),
    ('gpr-mgk', 'GaussianProcessRegressionUncertainty_MarginalizedGraphKernel_Config'),
    ('mlp-mve-morgan', 'MLP_Morgan_Regression_MVE_Config'),
    ('mlp-evi-morgan', 'MLP_Morgan_Regression_Evidential_Config'),
    ('dmpnn-rdkit-mve', 'DMPNN_RdkitNorm_Regression_MVE_Config'),
    ('dmpnn-rdkit-evi', 'DMPNN_RdkitNorm_Regression_Evidential_Config'),
])
@pytest.mark.parametrize('learning_type', ['explorative'])
@pytest.mark.parametrize('split_type', ['random'])
def test_ActiveLearning_PureCompound_Regression_Continue(dataset, modelset, learning_type, split_type):
    dataset, pure_columns, target_columns = dataset
    model_name, model_config = modelset
    model_config = '%s/../model_config/%s' % (CWD, model_config)
    save_dir = '%s/data/_%s_%s_%s_%s' % (CWD, dataset, model_name, learning_type, split_type)
    arguments = [
                    '--save_dir', save_dir,
                    '--data_path', '%s/data/%s.csv' % (CWD, dataset),
                    '--pure_columns'] + pure_columns + [
                    '--target_columns'] + target_columns + [
                    '--dataset_type', 'regression',
                    '--metrics', 'rmse', 'mae', 'r2',
                    '--learning_type', learning_type,
                    '--model_config_selector', model_config,
                    '--split_type', split_type,
                    '--split_sizes', '0.5', '0.5',
                    '--evaluate_stride', '1',
                    '--stop_ratio', '0.5'
                ]
    args = ActiveLearningArgs().parse_args(arguments)
    from scripts.ActiveLearning import main
    active_learner = main(args)
    assert int((active_learner.train_size + active_learner.pool_size) * 0.5) == active_learner.train_size
    # continue run
    arguments = [
                    '--save_dir', save_dir,
                    '--stop_ratio', '0.8'
                ]
    args = ActiveLearningContinueArgs().parse_args(arguments)
    from scripts.ALContinue import main
    active_learner = main(args)
    assert int((active_learner.train_size + active_learner.pool_size) * 0.8) == active_learner.train_size


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv'])
])
@pytest.mark.parametrize('modelset', [
    ('rf-morgan', 'RandomForest_Morgan_Config'),
])
@pytest.mark.parametrize('learning_type', ['explorative', 'exploitive', 'passive'])
@pytest.mark.parametrize('split_type', ['random', 'scaffold_random', 'scaffold_order'])
def test_ActiveLearning_PureCompound_Regression_1_Continue(dataset, modelset, learning_type, split_type):
    dataset, pure_columns, target_columns = dataset
    model_name, model_config = modelset
    model_config = '%s/../model_config/%s' % (CWD, model_config)
    save_dir = '%s/data/_%s_%s_%s_%s' % (CWD, dataset, model_name, learning_type, split_type)
    arguments = [
                    '--save_dir', save_dir,
                    '--data_path', '%s/data/%s.csv' % (CWD, dataset),
                    '--pure_columns'] + pure_columns + [
                    '--target_columns'] + target_columns + [
                    '--dataset_type', 'regression',
                    '--metrics', 'rmse', 'mae', 'r2',
                    '--learning_type', learning_type,
                    '--model_config_selector', model_config,
                    '--split_type', split_type,
                    '--split_sizes', '0.5', '0.5',
                    '--evaluate_stride', '1',
                    '--stop_ratio', '0.5'
                ]
    args = ActiveLearningArgs().parse_args(arguments)
    from scripts.ActiveLearning import main
    active_learner = main(args)
    assert int((active_learner.train_size + active_learner.pool_size) * 0.5) == active_learner.train_size
    # continue run
    arguments = [
                    '--save_dir', save_dir,
                    '--stop_ratio', '0.8'
                ]
    args = ActiveLearningContinueArgs().parse_args(arguments)
    from scripts.ALContinue import main
    active_learner = main(args)
    assert int((active_learner.train_size + active_learner.pool_size) * 0.8) == active_learner.train_size
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os

CWD = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.append('%s/..' % CWD)
from alb.args import ActiveLearningArgs
from ActiveLearning import main


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv'])
])
@pytest.mark.parametrize('modelset', [
    ('rf-morgan', 'RandomForest_Morgan_Config'),
    ('gpr-dpk-morgan', 'GaussianProcessRegressionUncertainty_DotProductKernel_Morgan_Config'),
    ('gpr-mgk', 'GaussianProcessRegressionUncertainty_MarginalizedGraphKernel_Config'),
    ('mlp-mve-morgan', 'MLP_Regression_MVE_Morgan_Config'),
    ('mlp-evi-morgan', 'MLP_Regression_Evidential_Morgan_Config'),
    ('dmpnn-rdkit-mve', 'DMPNN_RDKIT_Regression_MVE_Config'),
    ('dmpnn-rdkit-evi', 'DMPNN_RDKIT_Regression_Evidential_Config'),
])
@pytest.mark.parametrize('learning_type', ['explorative'])
@pytest.mark.parametrize('split_type', ['random'])
def test_ActiveLearning_PureCompound_Regression(dataset, modelset, learning_type, split_type):
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
    ]
    args = ActiveLearningArgs().parse_args(arguments)
    main(args)


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv'])
])
@pytest.mark.parametrize('modelset', [
    ('rf-morgan', 'RandomForest_Morgan_Config'),
])
@pytest.mark.parametrize('learning_type', ['explorative', 'exploitive', 'passive'])
@pytest.mark.parametrize('split_type', ['random', 'scaffold_random', 'scaffold_order'])
def test_ActiveLearning_PureCompound_Regression_1(dataset, modelset, learning_type, split_type):
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
                ]
    args = ActiveLearningArgs().parse_args(arguments)
    main(args)


@pytest.mark.parametrize('dataset', [
    ('bace', ['mol'], ['Class']),
])
@pytest.mark.parametrize('modelset', [
    ('rf-morgan', 'RandomForest_Morgan_Config'),
    ('lr-morgan', 'LogisticRegression_Morgan_Config'),
    ('gprv-dpk-morgan', 'GaussianProcessRegressionValue_DotProductKernel_Morgan_Config'),
    ('gpru-dpk-morgan', 'GaussianProcessRegressionUncertainty_DotProductKernel_Morgan_Config'),
    ('gpc-dpk-morgan', 'GaussianProcessClassification_DotProductKernel_Morgan_Config'),
    ('gprv-rbf-morgan', 'GaussianProcessRegressionValue_RBFKernel_RdkitNorm_Config'),
    ('gpru-rbf-morgan', 'GaussianProcessRegressionUncertainty_RBFKernel_RdkitNorm_Config'),
    ('gpc-rbf-morgan', 'GaussianProcessClassification_RBFKernel_RdkitNorm_Config'),
    ('gprv-mgk', 'GaussianProcessRegressionValue_MarginalizedGraphKernel_Config'),
    ('gpru-mgk', 'GaussianProcessRegressionUncertainty_MarginalizedGraphKernel_Config'),
    ('gpc-mgk', 'GaussianProcessClassification_MarginalizedGraphKernel_Config'),
    ('gpc-mgk-dpk-morgan', 'GaussianProcessClassification_MarginalizedGraphKernel_DotProductKernel_Morgan_Config'),
    ('gpc-mgk-rbf-rdkit', 'GaussianProcessClassification_MarginalizedGraphKernel_RBFKernel_RdkitNorm_Config'),
    ('mlp-morgan', 'MLP_BinaryClassification_Morgan_Config'),
    ('dmpnn', 'DMPNN_BinaryClassification_Config'),
    ('dmpnn-rdkit', 'DMPNN_RDKIT_BinaryClassification_Config'),
])
@pytest.mark.parametrize('learning_type', ['explorative'])
@pytest.mark.parametrize('split_type', ['random'])
def test_ActiveLearning_PureCompound_Binary(dataset, modelset, learning_type, split_type):
    dataset, pure_columns, target_columns = dataset
    model_name, model_config = modelset
    model_config = '%s/../model_config/%s' % (CWD, model_config)
    save_dir = '%s/data/_%s_%s_%s_%s' % (CWD, dataset, model_name, learning_type, split_type)
    arguments = [
                    '--save_dir', save_dir,
                    '--data_path', '%s/data/%s.csv' % (CWD, dataset),
                    '--pure_columns'] + pure_columns + [
                    '--target_columns'] + target_columns + [
                    '--dataset_type', 'classification',
                    '--metrics', 'roc-auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc',
                    '--learning_type', learning_type,
                    '--model_config_selector', model_config,
                    '--split_type', split_type,
                    '--split_sizes', '0.5', '0.5',
                    '--evaluate_stride', '1',
                ]
    args = ActiveLearningArgs().parse_args(arguments)
    main(args)


@pytest.mark.parametrize('dataset', [
    ('bace', ['mol'], ['Class']),
])
@pytest.mark.parametrize('modelset', [
    ('rf-morgan', 'RandomForest_Morgan_Config'),
])
@pytest.mark.parametrize('learning_type', ['explorative', 'exploitive', 'passive'])
@pytest.mark.parametrize('split_type', ['random', 'scaffold_random', 'scaffold_order'])
def test_ActiveLearning_PureCompound_Binary_1(dataset, modelset, learning_type, split_type):
    dataset, pure_columns, target_columns = dataset
    model_name, model_config = modelset
    model_config = '%s/../model_config/%s' % (CWD, model_config)
    save_dir = '%s/data/_%s_%s_%s_%s' % (CWD, dataset, model_name, learning_type, split_type)
    arguments = [
                    '--save_dir', save_dir,
                    '--data_path', '%s/data/%s.csv' % (CWD, dataset),
                    '--pure_columns'] + pure_columns + [
                    '--target_columns'] + target_columns + [
                    '--dataset_type', 'classification',
                    '--metrics', 'roc-auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc',
                    '--learning_type', learning_type,
                    '--model_config_selector', model_config,
                    '--split_type', split_type,
                    '--split_sizes', '0.5', '0.5',
                    '--evaluate_stride', '1',
                ]
    args = ActiveLearningArgs().parse_args(arguments)
    main(args)

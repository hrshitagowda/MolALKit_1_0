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
    ('np', ['drug', 'excp'], ['class'])
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
    ('gprv-mgk', 'GaussianProcessRegressionValue_MarginalizedGraphKernel2_Config'),
    ('gpru-mgk', 'GaussianProcessRegressionUncertainty_MarginalizedGraphKernel2_Config'),
    ('gpc-mgk', 'GaussianProcessClassification_MarginalizedGraphKernel2_Config'),
    ('gpc-mgk-dpk-morgan', 'GaussianProcessClassification_MarginalizedGraphKernel2_DotProductKernel_Morgan_Config'),
    ('gpc-mgk-rbf-rdkit', 'GaussianProcessClassification_MarginalizedGraphKernel2_RBFKernel_RdkitNorm_Config'),
    ('mlp-morgan', 'MLP_BinaryClassification_Morgan_Config'),
    ('dmpnn', 'DMPNN_BinaryClassification_Config'),
    ('dmpnn-rdkit', 'DMPNN_RDKIT_BinaryClassification_Config'),
])
@pytest.mark.parametrize('learning_type', ['explorative'])
@pytest.mark.parametrize('split_type', ['random'])
def test_ActiveLearning_PureCompound2_Binary(dataset, modelset, learning_type, split_type):
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
    ('np', ['pair'], ['class'])
])
@pytest.mark.parametrize('modelset', [
    ('rf-morgan-concat', 'RandomForest_Morgan_concat_Config'),
    ('rf-morgan-mean', 'RandomForest_Morgan_mean_Config'),
])
@pytest.mark.parametrize('learning_type', ['explorative', 'exploitive', 'passive'])
@pytest.mark.parametrize('split_type', ['random'])
def test_ActiveLearning_Mixture_Binary(dataset, modelset, learning_type, split_type):
    dataset, mixture_columns, target_columns = dataset
    model_name, model_config = modelset
    model_config = '%s/../model_config/%s' % (CWD, model_config)
    save_dir = '%s/data/_%s_%s_%s_%s' % (CWD, dataset, model_name, learning_type, split_type)
    arguments = [
        '--save_dir', save_dir,
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--mixture_columns'] + mixture_columns + [
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

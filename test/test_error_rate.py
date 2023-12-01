#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
import pandas as pd
from molalkit.args import ActiveLearningArgs
from model.test_model import run, al_results_check


CWD = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize('error_rate', ['0.1', '0.2', '0.3'])
def test_classification(error_rate):
    save_dir = os.path.join(CWD, 'test')
    arguments = [
        '--data_public', 'carcinogens_lagunin',
        '--metrics', 'roc-auc',
        '--learning_type', 'explorative',
        '--model_config_selector', 'RandomForest_RDKitNorm_Config',
        '--split_type', 'random',
        '--split_sizes', '0.5', '0.5',
        '--error_rate', error_rate,
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
    df1 = pd.read_csv('%s/train_init.csv' % save_dir)
    df2 = pd.read_csv('%s/pool_init.csv' % save_dir)
    df = pd.concat([df1, df2])
    df_ = pd.read_csv('%s/val.csv' % save_dir)
    assert abs(len(df) - len(df_)) <= 1
    assert df['flip_label'].sum() == int(float(error_rate) * len(df))

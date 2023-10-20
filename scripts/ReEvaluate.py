#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from tqdm import tqdm
from molalkit.args import ReEvaluateArgs
from molalkit.al.learner import eval_metric_func


def main(args: ReEvaluateArgs) -> None:
    args.logger.info('Start a new active learning run.')
    data_train = args.data_al_evaluator.copy()
    active_learning_traj_dict = {'training_size': []}
    for metric in args.metrics:
        active_learning_traj_dict[metric] = []
    n_list = list(range(args.evaluate_stride, len(args.data_al_evaluator), args.evaluate_stride))
    if n_list[-1] != len(data_train):
        n_list.append(len(data_train))
    for n in tqdm(n_list):
        active_learning_traj_dict['training_size'].append(n)
        data_train.data = args.data_al_evaluator.data[:n]
        args.model_evaluator.fit(data_train)
        y_pred = args.model_evaluator.predict_value(args.data_val_evaluator)
        for metric in args.metrics:
            metric_value = eval_metric_func(args.data_val_evaluator.y, y_pred, metric=metric)
            active_learning_traj_dict[metric].append(metric_value)
    pd.DataFrame(active_learning_traj_dict).to_csv(
        '%s/active_learning_extra_%d.traj' % (args.save_dir, args.evaluator_id), index=False)


if __name__ == '__main__':
    main(args=ReEvaluateArgs().parse_args())

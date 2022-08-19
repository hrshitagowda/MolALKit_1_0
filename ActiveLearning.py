#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from alb.args import ActiveLearningArgs
from alb.al.learner import ActiveLearner


def main(args: ActiveLearningArgs) -> None:
    # save args into a json file.
    args.save(os.path.join(args.save_dir, 'args.json'), skip_unpicklable=True)
    # active learning
    if args.yoked_learning:
        active_learner = ActiveLearner(save_dir=args.save_dir,
                                       dataset_type=args.dataset_type,
                                       metrics=args.metrics,
                                       learning_type=args.learning_type,
                                       model_selector=args.model_selector,
                                       model_evaluator=args.model_evaluator,
                                       dataset_train_selector=args.data_train_selector,
                                       dataset_pool_selector=args.data_pool_selector,
                                       dataset_val_evaluator=args.data_val_evaluator,
                                       dataset_train_evaluator=args.data_train_evaluator,
                                       dataset_pool_evaluator=args.data_pool_evaluator,
                                       evaluate_stride=args.evaluate_stride)
    else:
        active_learner = ActiveLearner(save_dir=args.save_dir,
                                       dataset_type=args.dataset_type,
                                       metrics=args.metrics,
                                       learning_type=args.learning_type,
                                       model_selector=args.model_selector,
                                       dataset_train_selector=args.data_train_selector,
                                       dataset_pool_selector=args.data_pool_selector,
                                       dataset_val_evaluator=args.data_val_evaluator,
                                       evaluate_stride=args.evaluate_stride)
    active_learner.run()


if __name__ == '__main__':
    main(args=ActiveLearningArgs().parse_args())

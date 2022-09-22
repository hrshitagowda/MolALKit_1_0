#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import time
from alb.args import ActiveLearningArgs
from alb.al.learner import ActiveLearner


def main(args: ActiveLearningArgs) -> ActiveLearner:
    # save args into a json file.
    # args.save(os.path.join(args.save_dir, 'args.json'), skip_unpicklable=True)
    # args.save(os.path.join(args.save_dir, 'args_test.json'), with_reproducibility=False, skip_unpicklable=True)
    # active learning
    args.logger.info('Start a new active learning run.')
    start = time.time()
    if args.load_checkpoint and os.path.exists('%s/al.pkl' % args.save_dir):
        args.logger.info('continue active learning using checkpoint file %s/al.pkl' % args.save_dir)
        active_learner = ActiveLearner.load(path=args.save_dir)
    elif args.yoked_learning:
        active_learner = ActiveLearner(save_dir=args.save_dir,
                                       dataset_type=args.dataset_type,
                                       metrics=args.metrics,
                                       learning_type=args.learning_type,
                                       batch_size=args.batch_size,
                                       stop_size=args.stop_size,
                                       model_selector=args.model_selector,
                                       model_evaluator=args.model_evaluator,
                                       dataset_train_selector=args.data_train_selector,
                                       dataset_pool_selector=args.data_pool_selector,
                                       dataset_val_evaluator=args.data_val_evaluator,
                                       dataset_train_evaluator=args.data_train_evaluator,
                                       dataset_pool_evaluator=args.data_pool_evaluator,
                                       model_extra_evaluators=args.model_extra_evaluators,
                                       dataset_train_extra_evaluators=args.data_train_extra_evaluators,
                                       dataset_pool_extra_evaluators=args.data_pool_extra_evaluators,
                                       dataset_val_extra_evaluators=args.data_val_extra_evaluators,
                                       evaluate_stride=args.evaluate_stride,
                                       extra_evaluators_only=args.extra_evaluators_only,
                                       save_cpt_stride=args.save_cpt_stride,
                                       logger=args.logger,
                                       seed=args.seed)
    else:
        active_learner = ActiveLearner(save_dir=args.save_dir,
                                       dataset_type=args.dataset_type,
                                       metrics=args.metrics,
                                       learning_type=args.learning_type,
                                       batch_size=args.batch_size,
                                       stop_size=args.stop_size,
                                       model_selector=args.model_selector,
                                       model_evaluator=None,
                                       dataset_train_selector=args.data_train_selector,
                                       dataset_pool_selector=args.data_pool_selector,
                                       dataset_val_evaluator=args.data_val_evaluator,
                                       dataset_train_evaluator=None,
                                       dataset_pool_evaluator=None,
                                       model_extra_evaluators=args.model_extra_evaluators,
                                       dataset_train_extra_evaluators=args.data_train_extra_evaluators,
                                       dataset_pool_extra_evaluators=args.data_pool_extra_evaluators,
                                       dataset_val_extra_evaluators=args.data_val_extra_evaluators,
                                       evaluate_stride=args.evaluate_stride,
                                       extra_evaluators_only=args.extra_evaluators_only,
                                       save_cpt_stride=args.save_cpt_stride,
                                       logger=args.logger,
                                       seed=args.seed)
    active_learner.run()
    end = time.time()
    args.logger.info('total time: %d s\n\n\n' % (end - start))
    return active_learner


if __name__ == '__main__':
    main(args=ActiveLearningArgs().parse_args())

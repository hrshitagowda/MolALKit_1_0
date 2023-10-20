#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import time
from molalkit.args import ActiveLearningArgs
from molalkit.al.learner import ActiveLearner


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
    else:
        active_learner = ActiveLearner(save_dir=args.save_dir,
                                       selection_method=args.selection_method,
                                       forgetter=args.forgetter,
                                       model_selector=args.model_selector,
                                       dataset_train_selector=args.data_train_selector,
                                       dataset_pool_selector=args.data_pool_selector,
                                       dataset_val_selector=args.data_val_selector,
                                       metrics=args.metrics,
                                       top_k_id=args.top_k_id,
                                       model_evaluators=args.model_evaluators,
                                       dataset_train_evaluators=args.data_train_evaluators,
                                       dataset_pool_evaluators=args.data_pool_evaluators,
                                       dataset_val_evaluators=args.data_val_evaluators,
                                       yoked_learning_only=args.yoked_learning_only,
                                       stop_size=args.stop_size,
                                       evaluate_stride=args.evaluate_stride,
                                       kernel=args.kernel_selector,
                                       save_cpt_stride=args.save_cpt_stride,
                                       seed=args.seed,
                                       logger=args.logger)
    active_learner.run(max_iter=args.max_iter)
    end = time.time()
    args.logger.info('total time: %d s\n\n\n' % (end - start))
    return active_learner


if __name__ == '__main__':
    main(args=ActiveLearningArgs().parse_args())

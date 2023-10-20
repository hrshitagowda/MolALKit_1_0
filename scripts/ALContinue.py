#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import time
from molalkit.args import ActiveLearningContinueArgs
from molalkit.al.learner import ActiveLearner


def main(args: ActiveLearningContinueArgs) -> ActiveLearner:
    # active learning
    args.logger.info('Start a new active learning run.')
    start = time.time()
    args.logger.info('continue active learning using checkpoint file %s/al.pkl' % args.save_dir)
    active_learner = ActiveLearner.load(path=args.save_dir)
    if args.stop_ratio is not None:
        if args.stop_size is None:
            args.stop_size = int(args.stop_ratio * (active_learner.train_size + active_learner.pool_size))
        else:
            args.stop_size = min(args.stop_size,
                                 int(args.stop_ratio * (active_learner.train_size + active_learner.pool_size)))
        assert args.stop_size >= 2
    active_learner.stop_size = args.stop_size
    active_learner.run()
    end = time.time()
    args.logger.info('total time: %d s\n\n\n' % (end - start))
    return active_learner


if __name__ == '__main__':
    main(args=ActiveLearningContinueArgs().parse_args())

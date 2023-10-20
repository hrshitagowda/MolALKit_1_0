#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        fh_v = logging.FileHandler(os.path.join(save_dir, '%s_verbose.log' % name))
        fh_v.setLevel(logging.DEBUG)
        # fh_q = logging.FileHandler(os.path.join(save_dir, '%s_quiet.log' % name))
        # fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        # logger.addHandler(fh_q)

    return logger

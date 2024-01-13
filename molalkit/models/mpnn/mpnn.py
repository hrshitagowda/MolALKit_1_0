#!/usr/bin/env python
# -*- coding: utf-8 -*-
from chemprop.models.mpnn import *
from chemprop.models.mpnn import MPNN as MPNNBase


class MPNN(MPNNBase):
    def __init__(self, continuous_fit, *args, **kwargs):
        super(MPNN, self).__init__(*args, **kwargs)
        self.continuous_fit = continuous_fit

    def fit_alb(self, train_data):
        if not self.continuous_fit and torch.cuda.is_available():
            torch.cuda.empty_cache()
        args = self.args
        args.train_data_size = len(train_data)
        logger = self.logger
        if logger is not None:
            debug, info = logger.debug, logger.info
        else:
            debug = info = print

        # Set pytorch seed for random initial weights
        torch.manual_seed(args.pytorch_seed)

        if args.dataset_type == 'classification':
            train_class_sizes = get_class_sizes(train_data, proportion=False)
            args.train_class_sizes = train_class_sizes

        if args.features_scaling:
            self.features_scaler = train_data.normalize_features(
                replace_nan_token=0)

        args.train_data_size = len(train_data)

        # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (
        # regression only)
        if args.dataset_type == 'regression':
            debug('Fitting scaler')
            scaler = train_data.normalize_targets()
            args.spectra_phase_mask = None
        else:
            args.spectra_phase_mask = None
            scaler = None

        # Get loss function
        loss_func = get_loss_func(args)
        """
        # Automatically determine whether to cache
        if len(train_data) <= args.cache_cutoff:
            set_cache_graph(True)
            num_workers = 0
        else:
            set_cache_graph(False)
            num_workers = args.num_workers
        """
        train_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            class_balance=args.class_balance,
            shuffle=True,
            seed=args.seed
        )

        if args.class_balance:
            debug(
                f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

        if self.continuous_fit and hasattr(self, 'models'):
            assert len(self.models) == args.ensemble_size
        else:
            self.models = []
            
        for model_idx in range(args.ensemble_size):
            # Tensorboard writer
            save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
            makedirs(save_dir)
            # try:
            #    writer = SummaryWriter(log_dir=save_dir)
            # except:
            #    writer = SummaryWriter(logdir=save_dir)
            writer = None
            if self.continuous_fit and len(self.models) == args.ensemble_size:
                debug(f'Loading model {model_idx} that fitted at previous iteration')
                model = self.models[model_idx]
            else:
                debug(f'Building model {model_idx} from scratch')
                model = MoleculeModel(args)
                if args.cuda:
                    info('Moving model to cuda')
                model = model.to(args.device)

            debug(model)

            # Optimizers
            optimizer = build_optimizer(model, args)

            # Learning rate schedulers
            scheduler = build_lr_scheduler(optimizer, args)

            n_iter = 0
            for epoch in trange(args.epochs):
                debug(f'Epoch {epoch}')
                n_iter = train(
                    model=model,
                    data_loader=train_data_loader,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    n_iter=n_iter,
                    logger=logger,
                    writer=writer
                )
                if isinstance(scheduler, ExponentialLR):
                    scheduler.step()
            if len(self.models) < args.ensemble_size:
                assert len(self.models) == model_idx
                self.models.append(model)
                
            self.scaler = scaler

# using this config as the configuration file
config = {
    'average_reset_epoch_interval': 20,
    'distributed_backend': 'gloo',
    'n_workers': 10,
    'num_epochs': 30,
    'checkpoints': [],
    'optimizer_batch_size': 10,  # eatch worker

    'optimizer_learning_rate': 0.01,  # Tuned for batch size 128 (single worker)
    'optimizer_decay_at_epochs': [20, 40],
    'optimizer_decay_with_factor': 10.0,
    'optimizer_scale_lr_with_factor': None,  # set to override world_size as a factor
    'optimizer_scale_lr_with_warmup_epochs': False,  # scale lr by world size

    'optimizer_memory': True,

    'task': 'Cifar',
    'dataset_name': 'Cifar10',
    'task_architecture': 'ResNet18',
    'optimizer_reducer': 'SASGReducer',
    'optimizer_reducer_compression': 0.01,
    'log_verbosity': 1,
    'local_rank': 0,
    'seed': 42,
    'rank': 0,

    'using_lag': False,
    'D': 10
    }

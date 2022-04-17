#!/usr/bin/env python3

import datetime
import os
import re
import time

import numpy as np
import torch

import gradient_reducers
import tasks
from mean_accumulator import MeanAccumulator
from timer import Timer
import argparse
from config import config
import random
import torch.distributed as dist
import time

NS = 1.0 / 1_000_000_000

output_dir = "/gf3/home/dxg/sasg-dis/output.tmp"


def main():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    if torch.distributed.is_available():

        torch.distributed.init_process_group(
            backend=config["distributed_backend"],
            timeout=datetime.timedelta(seconds=120),
        )
        config["n_workers"] = torch.distributed.get_world_size()
        config["rank"] = torch.distributed.get_rank()
        print(
            "Distributed init: rank {}/{}".format(
                config["rank"], config["n_workers"]
            )
        )
    device = torch.device("cuda:" + str(config["local_rank"]) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(config["local_rank"])
    seed_torch(config["seed"] + config["rank"])

    timer = Timer(verbosity_level=config["log_verbosity"], log_fn=metric)
    task = tasks.build(task_name=config["task"], device=device, timer=timer, **config)
    print("Rank {} build task {} on {} ".format(config["rank"], config["task"], config["dataset_name"]))

    reducer = get_reducer(device, timer)
    print("Rank {} get reducer".format(config["rank"]))

    para_D_list = []
    grad_workers = [[] for col in range(2 * config["n_workers"])]
    bits_communicated = 0
    tau = 0
    memories = [torch.zeros_like(param) for param in task.state]
    send_buffers = [torch.zeros_like(param) for param in task.state]
    param_for_thread_old = [param.data.clone().detach() for param in task.state]

    for epoch in range(config["num_epochs"]):
        epoch_metrics = MeanAccumulator()
        train_loss, num_loss = 0.0, 0
        train_loader = task.train_iterator(config["optimizer_batch_size"])

        for i, batch in enumerate(train_loader):
            epoch_frac = epoch + i / len(train_loader)
            train_idx = epoch * len(train_loader) + i
            skip_num_batch = 0
            lr, lr_thread = get_learning_rate(epoch_frac)

            with timer("batch", epoch_frac):
                _, grads, grads_new, grads_old, metrics = task.batch_loss_and_gradient(train_idx, batch)
                epoch_metrics.add(metrics)
                train_loss += metrics["cross_entropy"].item()
                num_loss += 1

                if config["using_lag"] and train_idx >= config["D"]:
                    thread = np.sum(para_D_list)
                    thread = 1.0 * thread / (lr_thread * (config["n_workers"] ** 2))

                    if compute_thread(grads_new, grads_old) > thread or tau > config["D"]:
                        flag_lag = torch.tensor(1, dtype=torch.uint8, device=device)
                        tau = 0
                    else:
                        flag_lag = torch.tensor(0, dtype=torch.uint8, device=device)
                        skip_num_batch += 1
                        tau += 1
                else:
                    flag_lag = torch.tensor(1, dtype=torch.uint8, device=device)

                with timer("batch.accumulate", epoch_frac, verbosity=2):
                    for grad, memory, send_bfr in zip(grads, memories, send_buffers):
                        if config["optimizer_memory"]:
                            send_bfr.data[:] = lr * grad + memory
                        else:
                            send_bfr.data[:] = lr * grad

                with timer("batch.reduce", epoch_frac):
                    """
                    Reduce gradients between the workers in place
                    :param flag_lag -> communication flag_lag
                    :param send_buffers -> grad_in: dictionary
                    :param grads -> grad_out: dictionary
                    :param memories -> memory_out: dictionary
                    :param grad_workers -> grads for lag: dictionary
                    """
                    bits_uplink, comm_time_batch = reducer.reduce(flag_lag, send_buffers, grads, memories, grad_workers)
                    bits_communicated += bits_uplink

                with timer("batch.step", epoch_frac, verbosity=2):
                    for param, grad in zip(task.state, grads):
                        param.data.add_(grad, alpha=-1)

                param_for_thread_new = [parameter.data.clone().detach() for parameter in task.model.parameters()]
                with timer("batch.compute_thread2", epoch_frac):
                    thread_d = compute_thread(param_for_thread_new, param_for_thread_old)
                para_D_list.insert(0, thread_d.item())
                if len(para_D_list) >= config["D"]:
                    para_D_list.pop()
                param_for_thread_old = param_for_thread_new

        with timer("epoch_metrics.collect", epoch + 1.0, verbosity=2):
            epoch_metrics.reduce()
            for key, value in epoch_metrics.value().items():
                metric(
                    key,
                    {"value": value.item(), "epoch": epoch + 1.0, "bits": bits_communicated},
                    tags={"split": "train"},
                )

        with timer("test.last", epoch, verbosity=2):
            test_stats = task.test()
            for key, value in test_stats.items():
                metric(
                    f"last_{key}",
                    {"value": value.item(), "epoch": epoch + 1.0, "bits": bits_communicated},
                    tags={"split": "test"},
                )

        if epoch in config["checkpoints"] and torch.distributed.get_rank() == 0:
            with timer("checkpointing"):
                save(
                    os.path.join(output_dir, "epoch_{:03d}".format(epoch)),
                    task.state_dict(),
                    epoch + 1.0,
                    test_stats,
                )
                # Save running average model @TODO
        print(timer.summary())

        if config["rank"] == 0:
            timer.save_summary(os.path.join(output_dir, config["optimizer_reducer"]+"-timer_summary.json"))


def save(destination_path, model_state, epoch, test_stats):
    """Save a checkpoint to disk"""
    # Workaround for RuntimeError("Unknown Error -1")
    # https://github.com/pytorch/pytorch/issues/10577
    time.sleep(1)

    torch.save(
        {"epoch": epoch, "test_stats": test_stats, "model_state_dict": model_state},
        destination_path,
    )


def get_learning_rate(epoch):
    """Apply any learning rate schedule"""
    lr = config["optimizer_learning_rate"]
    lr_thread = config["optimizer_learning_rate"]

    for decay_epoch in config["optimizer_decay_at_epochs"]:
        if epoch >= decay_epoch:
            lr /= config["optimizer_decay_with_factor"]

        if epoch >= decay_epoch + 200/500:
            lr_thread /= config["optimizer_decay_with_factor"]
    return lr, lr_thread


def get_reducer(device, timer):
    """Configure the reducer from the config"""
    if config["optimizer_reducer"] == "LASGReducer":
        config["using_lag"] = True
        if config["rank"] == 0:
            print("*** LASG method ***")
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer, 
        )
    elif config["optimizer_reducer"] == "ExactReducer":
        config["using_lag"] = False
        if config["rank"] == 0:
            print("*** SGD method ***")
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer
        )

    elif config["optimizer_reducer"] == "SASGReducer":
        config["using_lag"] = True
        if config["rank"] == 0:
            print("*** SASG method ***")
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            compression=config["optimizer_reducer_compression"],
        )
    elif config["optimizer_reducer"] == "TopKReducer":
        config["using_lag"] = False
        if config["rank"] == 0:
            print("*** TopK method ***")
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            compression=config["optimizer_reducer_compression"],
        )
    else:
        raise ValueError("Unknown reducer type")


@torch.jit.script
def l2norm(tensor):
    """Compute the L2 Norm of a tensor in a fast and correct way"""
    return torch.sqrt(torch.sum(tensor ** 2))


def log_info(info_dict):
    """Add any information to MongoDB
       This function will be overwritten when called through run.py"""
    pass


def log_metric(name, values, tags={}):
    """Log timeseries data
       This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
    print("{name:30s} - {values} ({tags})".format(name=name, values=values, tags=tags))


def info(*args, **kwargs):
    if config["rank"] == 0:
        log_info(*args, **kwargs)


def metric(*args, **kwargs):
    if config["rank"] == 0:
        log_metric(*args, **kwargs)


def compute_thread(old, new):
    result = 0.0
    for i in range(len(old)):
        result += torch.sum((old[i] - new[i]) ** 2)
    return result


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)

    parser.add_argument("--architecture", type=str, default="ResNet18")
    parser.add_argument("--dataset", type=str, default="Cifar10")
    parser.add_argument("--reducer", type=str, default="SASGReducer")

    args = parser.parse_args()

    config["task_architecture"] = args.architecture
    config["dataset_name"] = args.dataset
    config["optimizer_reducer"] = args.reducer

    config["local_rank"] = args.local_rank
    os.environ["CUDA_DEVICE"] = str(config["local_rank"])

    main()


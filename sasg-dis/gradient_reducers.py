import datetime
from dis import dis
import os
import time
from contextlib import contextmanager
from turtle import up
from typing import List
import torch.distributed as dist

import numpy as np
import torch

NS = 1.0 / 1_000_000_000  # 1[ns] in [s]


class Reducer:
    def __init__(self, random_seed, device, timer):
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device
        self.timer = timer

    def reduce(self, flag_lag, grad_in, grad_out, memory_out, grad_workers):
        """Return communicated bits"""
        raise NotImplementedError()


class SASGReducer(Reducer):
    """
    Use same amount as rank-based
    """
    def __init__(self, random_seed, device, timer, compression=0.01):
        super().__init__(random_seed, device, timer)
        self.compression = compression  # value and position

    def reduce(self, flag_lag, grad_in, grad_out, memory_out, grad_workers):
        """
        Reduce gradients between the workers in place
        :param flag_lag: bool->communication flag_lag
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        :param grad_workers: dictionary -> grads for lag
        """
        bits_communicated = 0
        uplink_time = 0.0
        flatgrad_size = 0
        flattopk_size = 0
        grad_idx = [0]
        topk_idx = [0]
        for tensor in grad_in:
            flatgrad_size += tensor.nelement()
            grad_idx.append(grad_idx[-1] + tensor.nelement())
            top_size = max(1, int(self.compression * tensor.nelement()))
            flattopk_size += top_size
            topk_idx.append(topk_idx[-1] + top_size)
        flatgrad_start_idx = grad_idx[:-1]
        flatgrad_end_idx = grad_idx[1:]
        flattopk_start_idx = topk_idx[:-1]
        flattopk_end_idx = topk_idx[1:]

        flat_values = torch.empty(flattopk_size)  # topk size
        flat_positions = torch.empty(flattopk_size, dtype=torch.int)  # topk size
        grads_mean = torch.zeros(flatgrad_size, device=self.device)  # grad size

        if flag_lag:
            for tensor, start, end in zip(grad_in, flattopk_start_idx, flattopk_end_idx):
                top_size = max(1, int(self.compression * tensor.nelement()))
                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
                values = tensor.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions

            for tensor, mem, start, end in zip(
                grad_in, memory_out, flattopk_start_idx, flattopk_end_idx
            ):
                positions = flat_positions[start:end]
                mem.data[:] = tensor
                mem.view(-1)[positions.long()] = 0.0

        if self.n_workers > 1:
            if self.rank == 0:
                flag_list = [torch.empty_like(flag_lag) for i in range(self.n_workers)]
                dist.gather(flag_lag, gather_list=flag_list, dst=0)

                torch.cuda.synchronize()
                start_time = time.time_ns() * NS
                with self.timer("comm_server.uplink", verbosity=1):
                    if flag_lag:
                        grad_workers[0] = flat_values
                        grad_workers[self.n_workers] = flat_positions
                    else:
                        flat_values = grad_workers[0]
                        flat_positions = grad_workers[self.n_workers]
                    for start_all, start, end in zip(flatgrad_start_idx, flattopk_start_idx, flattopk_end_idx):
                        positions = flat_positions[start:end] + start_all
                        values = flat_values[start:end]
                        grads_mean[positions.long()] += values.to(self.device)

                    for i in range(self.n_workers-1):
                        comm_flag = flag_list[i+1]
                        if comm_flag:
                            grad_values = torch.empty_like(flat_values)
                            dist.recv(tensor=grad_values, src=i+1, tag=11)
                            grad_positions = torch.empty_like(flat_values, dtype=torch.int)
                            dist.recv(tensor=grad_positions, src=i+1, tag=12)
                            grad_workers[i+1] = grad_values
                            grad_workers[i+1+self.n_workers] = grad_positions
                        else:
                            grad_values = grad_workers[i+1]
                            grad_positions = grad_workers[i+1+self.n_workers]
                        for start_all, start, end in zip(flatgrad_start_idx, flattopk_start_idx, flattopk_end_idx):
                            positions = grad_positions[start:end] + start_all
                            values = grad_values[start:end]
                            grads_mean[positions.long()] += values.to(self.device)

                torch.cuda.synchronize()
                end_time = time.time_ns() * NS
                uplink_time = end_time - start_time

                dist.broadcast(grads_mean, src=0)
            else:
                dist.gather(flag_lag, gather_list=None, dst=0)
                torch.cuda.synchronize()
                start_time = time.time_ns() * NS
                with self.timer("comm_worker.uplink", verbosity=1):
                    if flag_lag:
                        dist.send(tensor=flat_values, dst=0, tag=11)
                        dist.send(tensor=flat_positions, dst=0, tag=12)
                
                torch.cuda.synchronize()
                end_time = time.time_ns() * NS
                uplink_time = end_time - start_time
                dist.broadcast(grads_mean, src=0)
        else:
            for start_all, start, end in zip(flatgrad_start_idx, flattopk_start_idx, flattopk_end_idx):
                positions = flat_positions[start:end] + start_all
                values = flat_values[start:end]
                grads_mean[positions.long()] += values.to(self.device)
        if flag_lag:
            bits_communicated += n_bits(flat_values) + n_bits(flat_positions)

        for out, start, end in zip(grad_out, flatgrad_start_idx, flatgrad_end_idx):
            out.data[:] = grads_mean[start:end].reshape_as(out)

        return bits_communicated, uplink_time


class TopKReducer(Reducer):
    """
    Use same amount as rank-based
    """
    def __init__(self, random_seed, device, timer, compression=0.01):
        super().__init__(random_seed, device, timer)
        self.compression = compression

    def reduce(self, flag_lag, grad_in, grad_out, memory_out, grad_workers):
        """
        Reduce gradients between the workers in place
        :param flag_lag: bool->communication flag_lag
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        :param grad_workers: dictionary -> grads for lag
        """
        bits_communicated = 0
        uplink_time = 0.0

        flatgrad_size = 0
        flattopk_size = 0
        grad_idx = [0]
        topk_idx = [0]
        for tensor in grad_in:
            flatgrad_size += tensor.nelement()
            grad_idx.append(grad_idx[-1] + tensor.nelement())
            top_size = max(1, int(self.compression * tensor.nelement()))
            flattopk_size += top_size
            topk_idx.append(topk_idx[-1] + top_size)
        flatgrad_start_idx = grad_idx[:-1]
        flatgrad_end_idx = grad_idx[1:]
        flattopk_start_idx = topk_idx[:-1]
        flattopk_end_idx = topk_idx[1:]

        flat_values = torch.empty(flattopk_size)  # topk size
        flat_positions = torch.empty(flattopk_size, dtype=torch.int)  # topk size
        grads_mean = torch.zeros(flatgrad_size, device=self.device)  # grad size

        for tensor, start, end in zip(grad_in, flattopk_start_idx, flattopk_end_idx):
            top_size = max(1, int(self.compression * tensor.nelement()))
            _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
            values = tensor.view(-1)[positions].contiguous()
            flat_values[start:end] = values
            flat_positions[start:end] = positions

        for tensor, mem, start, end in zip(
            grad_in, memory_out, flattopk_start_idx, flattopk_end_idx
        ):
            positions = flat_positions[start:end]
            mem.data[:] = tensor
            mem.view(-1)[positions.long()] = 0.0

        if self.n_workers > 1:
            if self.rank == 0:
                flag_list = [torch.empty_like(flag_lag) for i in range(self.n_workers)]
                dist.gather(flag_lag, gather_list=flag_list, dst=0)

                torch.cuda.synchronize()
                start_time = time.time_ns() * NS
                with self.timer("comm_server.uplink", verbosity=1):
                    for start_all, start, end in zip(flatgrad_start_idx, flattopk_start_idx, flattopk_end_idx):
                        positions = flat_positions[start:end] + start_all
                        values = flat_values[start:end]
                        grads_mean[positions.long()] += values.to(self.device)
                    for i in range(self.n_workers-1):
                        grad_values = torch.empty_like(flat_values)
                        dist.recv(tensor=grad_values, src=i+1, tag=11)
                        grad_positions = torch.empty_like(flat_values, dtype=torch.int)
                        dist.recv(tensor=grad_positions, src=i+1, tag=12)
                        for start_all, start, end in zip(flatgrad_start_idx, flattopk_start_idx, flattopk_end_idx):
                            positions = grad_positions[start:end] + start_all
                            values = grad_values[start:end]
                            grads_mean[positions.long()] += values.to(self.device)
                torch.cuda.synchronize()
                end_time = time.time_ns() * NS
                uplink_time = end_time - start_time
                dist.broadcast(grads_mean, src=0)
            else:
                dist.gather(flag_lag, gather_list=None, dst=0)
                
                torch.cuda.synchronize()
                start_time = time.time_ns() * NS
                with self.timer("comm_worker.uplink", verbosity=1):
                    dist.send(tensor=flat_values, dst=0, tag=11)
                    dist.send(tensor=flat_positions, dst=0, tag=12)
                torch.cuda.synchronize()
                end_time = time.time_ns() * NS
                uplink_time = end_time - start_time
                dist.broadcast(grads_mean, src=0)
        else:
            for start_all, start, end in zip(flatgrad_start_idx, flattopk_start_idx, flattopk_end_idx):
                positions = flat_positions[start:end] + start_all
                values = flat_values[start:end]
                grads_mean[positions.long()] += values.to(self.device)
        bits_communicated += n_bits(flat_values) + n_bits(flat_positions)

        for out, start, end in zip(grad_out, flatgrad_start_idx, flatgrad_end_idx):
            out.data[:] = grads_mean[start:end].reshape_as(out)

        return bits_communicated, uplink_time


class LASGReducer(Reducer):
    def reduce(self, flag_lag, grad_in, grad_out, memory_out, grad_workers):
        """
        Reduce gradients between the workers in place
        :param flag_lag: bool->communication flag_lag
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        :param grad_workers: dictionary -> grads for lag
        """
        bits_communicated = 0
        uplink_time = 0.0
        for mem in memory_out:
            mem.zero_()
        flatgrad_size = 0
        tensor_idx = [0]
        for tensor in grad_in:
            flatgrad_size += tensor.nelement()
            tensor_idx.append(tensor_idx[-1] + tensor.nelement())
        flatgrad_start_idx = tensor_idx[:-1]
        flatgrad_end_idx = tensor_idx[1:]
        flat_values = torch.empty(flatgrad_size)

        if flag_lag:
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                flat_values[start:end] = tensor.view(-1)  # flat_value on cpu

        grads_mean = torch.zeros(flatgrad_size, device=self.device)

        if self.n_workers > 1:
            if self.rank == 0:
                flag_list = [torch.empty_like(flag_lag) for i in range(self.n_workers)]
                dist.gather(flag_lag, gather_list=flag_list, dst=0)

                torch.cuda.synchronize()
                start_time = time.time_ns() * NS
                with self.timer("comm_server.uplink", verbosity=1):
                    if flag_lag:
                        grad_workers[0] = flat_values
                    else:
                        flat_values = grad_workers[0]
                    grads_mean += flat_values.to(self.device)
                    for i in range(self.n_workers-1):
                        comm_flag = flag_list[i+1]
                        if comm_flag:
                            grad_value = torch.empty_like(flat_values)
                            dist.recv(tensor=grad_value, src=i+1, tag=11)
                            grad_workers[i+1] = grad_value
                        else:
                            grad_value = grad_workers[i+1]
                        grads_mean += grad_value.to(self.device)
                torch.cuda.synchronize()
                end_time = time.time_ns() * NS
                uplink_time = end_time - start_time
                dist.broadcast(grads_mean, src=0)
            else:
                dist.gather(flag_lag, gather_list=None, dst=0)

                torch.cuda.synchronize()
                start_time = time.time_ns() * NS
                with self.timer("comm_worker.uplink", verbosity=1):
                    if flag_lag:
                        dist.send(tensor=flat_values, dst=0, tag=11)
                torch.cuda.synchronize()
                end_time = time.time_ns() * NS
                uplink_time = end_time - start_time
                dist.broadcast(grads_mean, src=0)
        else:
            grads_mean += flat_values.to(self.device)
        if flag_lag:
            bits_communicated += n_bits(flat_values)

        for out, start, end in zip(grad_out, flatgrad_start_idx, flatgrad_end_idx):
            out.data[:] = grads_mean[start:end].reshape_as(out)

        return bits_communicated, uplink_time


class ExactReducer(Reducer):
    def reduce(self, flag_lag, grad_in, grad_out, memory_out, grad_workers):
        """
        Reduce gradients between the workers in place
        :param flag_lag: bool->communication flag_lag
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        :param grad_workers: dictionary -> grads for lag
        """
        bits_communicated = 0
        uplink_time = 0.0

        for mem in memory_out:
            mem.zero_()
        flatgrad_size = 0
        tensor_idx = [0]
        for tensor in grad_in:
            flatgrad_size += tensor.nelement()
            tensor_idx.append(tensor_idx[-1] + tensor.nelement())
        flatgrad_start_idx = tensor_idx[:-1]
        flatgrad_end_idx = tensor_idx[1:]
        flat_values = torch.empty(flatgrad_size)

        for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
            flat_values[start:end] = tensor.view(-1)  # flat_value on cpu

        grads_mean = torch.zeros(flatgrad_size, device=self.device)

        if self.n_workers > 1:
            if self.rank == 0:
                flag_list = [torch.empty_like(flag_lag) for i in range(self.n_workers)]
                dist.gather(flag_lag, gather_list=flag_list, dst=0)

                torch.cuda.synchronize()
                start_time = time.time_ns() * NS
                with self.timer("comm_server.uplink", verbosity=1):
                    grads_mean += flat_values.to(self.device)

                    for i in range(self.n_workers-1):
                        grad_value = torch.empty_like(flat_values)
                        dist.recv(tensor=grad_value, src=i+1, tag=11)
                        grads_mean += grad_value.to(self.device)

                torch.cuda.synchronize()
                end_time = time.time_ns() * NS
                uplink_time = end_time - start_time

                dist.broadcast(grads_mean, src=0)
            else:
                dist.gather(flag_lag, gather_list=None, dst=0)
                torch.cuda.synchronize()
                start_time = time.time_ns() * NS
                with self.timer("comm_worker.uplink", verbosity=1):
                    dist.send(tensor=flat_values, dst=0, tag=11)

                torch.cuda.synchronize()
                end_time = time.time_ns() * NS
                uplink_time = end_time - start_time

                dist.broadcast(grads_mean, src=0)
        else:
            grads_mean += flat_values.to(self.device)

        bits_communicated += n_bits(flat_values)

        for out, start, end in zip(grad_out, flatgrad_start_idx, flatgrad_end_idx):
            out.data[:] = grads_mean[start:end].reshape_as(out)

        return bits_communicated, uplink_time
    

@torch.jit.script
def l2norm(x):
    return torch.sqrt(torch.sum(x ** 2))


def normalize_(tensor):
    """Divide by L2 norm. In place"""
    tensor /= l2norm(tensor)


def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()

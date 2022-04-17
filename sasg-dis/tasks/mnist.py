import os
from copy import deepcopy
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader
import torchvision

from mean_accumulator import MeanAccumulator
from .utils import DistributedSampler
from . import cifar_architectures
from config import config


class Batch:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class MNISTTask:
    def __init__(self, device, timer, architecture, seed):
        self._device = device
        self._timer = timer
        self._seed = seed
        self._architecture = architecture
        self._train_set, self._test_set = self._create_dataset(
        )

        self.model = self._create_model()
        self.model_lag = self._create_model()
        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)

        self._epoch = 0  # Counts how many times train_iterator was called

        self.lag_state_dict = deepcopy(self.model.state_dict())
        self.state = [parameter for parameter in self.model.parameters()]
        self.buffers = [buffer for buffer in self.model.buffers()]
        self.parameter_names = [name for (name, _) in self.model.named_parameters()]

    def train_iterator(self, batch_size: int) -> Iterable[Batch]:
        sampler = DistributedSampler(dataset=self._train_set, add_extra_samples=True)
        sampler.set_epoch(self._epoch)

        train_loader = DataLoader(
            self._train_set,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=2,
        )

        self._epoch += 1

        return BatchLoader(train_loader, self._device)

    def batch_loss_and_gradient(
        self, idx: int, batch: Batch
    ) -> Tuple[float, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, float]]:
        self._zero_grad()
        if idx < config['D']:
            with self._timer("batch.forward", float(self._epoch)):
                prediction = self.model(batch.x)
                f = self._criterion(prediction, batch.y)
            with self._timer("batch.backward", float(self._epoch)):
                f.backward()
            with self._timer("batch.evaluate", float(self._epoch)):
                metrics = self.evaluate_prediction(prediction, batch.y)
            df = [parameter.grad for parameter in self.model.parameters()]
            grads_new = [parameter.grad.clone().detach() for parameter in self.model.parameters()]
            grads_old = grads_new
            self.lag_state_dict = deepcopy(self.model.state_dict())
        else:
            self.model_lag.load_state_dict(self.lag_state_dict)
            with self._timer("batch.forward", float(self._epoch)):
                prediction = self.model(batch.x)
                f = self._criterion(prediction, batch.y)
                prediction_lag = self.model_lag(batch.x)
                f_lag = self._criterion(prediction_lag, batch.y)
            with self._timer("batch.backward", float(self._epoch)):
                f.backward()
                f_lag.backward()
            with self._timer("batch.evaluate", float(self._epoch)):
                metrics = self.evaluate_prediction(prediction, batch.y)
            df = [parameter.grad for parameter in self.model.parameters()]
            grads_new = [parameter.grad.clone().detach() for parameter in self.model.parameters()]
            grads_old = [parameter.grad.clone().detach() for parameter in self.model_lag.parameters()]
            self.lag_state_dict = deepcopy(self.model.state_dict())

        return f.detach(), df, grads_new, grads_old, metrics

    def evaluate_prediction(self, model_output, reference):
        """
        Compute a series of scalar loss values for a predicted batch and references
        """
        with torch.no_grad():
            _, top5 = model_output.topk(5)
            top1 = top5[:, 0]
            cross_entropy = self._criterion(model_output, reference)
            accuracy = top1.eq(reference).sum().float() / len(reference)
            top5_accuracy = reference.unsqueeze(1).eq(top5).sum().float() / len(reference)
            return {
                "cross_entropy": cross_entropy.detach(),
                "accuracy": accuracy.detach(),
                "top5_accuracy": top5_accuracy.detach(),
            }

    def test(self, state_dict=None) -> float:
        """
        Compute the average loss on the test set.
        The task is completed as soon as the output is below self.target_test_loss.
        If the model has batch normalization or dropout, this will run in eval mode.
        """
        test_loader = BatchLoader(
            DataLoader(
                self._test_set,
                batch_size=250,
                num_workers=1,
                drop_last=False,
                pin_memory=True,
                sampler=DistributedSampler(dataset=self._test_set, add_extra_samples=False),
            ),
            self._device,
        )

        if state_dict:
            test_model = self._create_test_model(state_dict)
        else:
            test_model = self.model
            test_model.eval()

        mean_metrics = MeanAccumulator()

        for batch in test_loader:
            with torch.no_grad():
                prediction = test_model(batch.x)
                metrics = self.evaluate_prediction(prediction, batch.y)
            mean_metrics.add(metrics)
        mean_metrics.reduce()  # Collect over workers

        test_model.train()
        return mean_metrics.value()

    def state_dict(self):
        """Dictionary containing the model state (buffers + tensors)"""
        return self.model.state_dict()

    def _create_model(self):
        """Create a PyTorch module for the model"""
        torch.random.manual_seed(self._seed)
        model = getattr(cifar_architectures, self._architecture)()
        model.to(self._device)
        model.train()
        return model

    def _create_test_model(self, state_dict):
        test_model = deepcopy(self.model)
        test_model.load_state_dict(state_dict)
        test_model.eval()
        return test_model

    def _create_dataset(self, data_root="/gf3/home/dxg/sasg-dis/data"):
        """Create train and test datasets"""

        dataset = torchvision.datasets.MNIST
        
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        training_set = dataset(root=data_root, train=True, download=True, transform=transforms)
        test_set = dataset(root=data_root, train=False, download=True, transform=transforms)

        return training_set, test_set

    def _zero_grad(self):
        self.model.zero_grad()
        self.model_lag.zero_grad()


class BatchLoader:
    """
    Utility that transforms a DataLoader that is an iterable over (x, y) tuples
    into an iterable over Batch() tuples, where its contents are already moved
    to the selected device.
    """

    def __init__(self, dataloader, device):
        self._dataloader = dataloader
        self._device = device

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        for x, y in self._dataloader:
            x = x.to(self._device)
            y = y.to(self._device)
            yield Batch(x, y)

## Communication-Efficient Distributed Learning via Sparse and Adaptive Gradient Descent

### Environment

```
# GPU environment required
torch>=1.10.0
torchvision>=0.11.1
numpy>=1.19.5
```

### Dataset

The MNIST, CIFAR-10, and CIFAR-100 datasets can be downloaded automatically by `torchvision.datasets`.


### Example Usage

```
python -m torch.distributed.launch --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
                                   --node_rank $NODE_RANK --master_addr $MASTER_ADDR \
                                   --master_port $MASTER_PORT train.py \
                                   --architecture ResNet18 --dataset Cifar100 \
                                   --reducer SASGReducer  
```


### Usage

```
usage: train.py [-h] [--local_rank LOCAL_RANK] [--lr LR]
                [--batchsize BATCHSIZE] [--lr_interval LR_INTERVAL] 
                [--epoch EPOCH] [--seed SEED]
                [--model {mnistnet,resnet18}]
                [--dataset {mnist,cifar10,cifar100}]
                [--optimizer_reducer {sasg,lasg,sparse,sgd}]

optional arguments:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
  --lr LR               training learning rate.
  --batchsize BATCHSIZE
                        training and eval batch size.
  --lr_interval LR_INTERVAL
                        learning rate update interval.
  --epoch EPOCH         number of train epochs.
  --seed SEED           random seed.
  --model {mnistnet,resnet}
                        model architecture.
  --dataset {mnist,cifar10,cifar100}
                        dataset name and dataset path, The cifar10, cifar100,
                        and mnist datasets can be downloaded automatically by
                        torchvision.datasets.
  --optimizer_reducer {SASGReducer,LASGReducer,TopKReducer,ExactReducer}
  Corresponding four algorithms: sasg, lasg, sparse, sgd
```



#### Note

* The source code for the distributed experiments is in the `sasg_dis` folder.
* The `.ipynb` file presents the results of the simulation experiments.
* The results of the experiments for different random seeds are contained in the `log_seed` folder.

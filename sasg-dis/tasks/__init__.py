def build(task_name, seed, device, timer, **kwargs):
    if task_name == "Cifar":
        from .cifar import CifarTask

        return CifarTask(
            seed=seed,
            device=device,
            timer=timer,
            architecture=kwargs.get("task_architecture", "ResNet18"),
        )
    elif task_name == "MNIST":
        from .mnist import MNISTTask

        return MNISTTask(
            seed=seed,
            device=device,
            timer=timer,
            architecture=kwargs.get("task_architecture", "MNISTNet"),
        )
    else:
        raise ValueError("Unknown task name")

_base_ = ["./pipelines/fixmatch_pipeline.py"]

__train_pipeline = {{_base_.train_pipeline}}
__test_pipeline = {{_base_.test_pipeline}}

seed = 1234

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    num_classes=100,
    train=dict(
        type="TVDatasetSplit",
        base="CIFAR100",
        num_classes=100,
        train=True,
        data_prefix="data/torchvision/cifar100",
        num_images=400,
        pipeline=__train_pipeline,
        samples_per_gpu=16,
        workers_per_gpu=4,
        seed=seed,
        download=True,
    ),
    val=dict(
        type="TVDatasetSplit",
        base="CIFAR100",
        num_classes=100,
        train=True,
        data_prefix="data/torchvision/cifar100",
        num_images=10000,
        samples_per_gpu=128,
        workers_per_gpu=4,
        seed=seed,
        pipeline=__test_pipeline,
        download=True,
    ),
    test=dict(
        type="TVDatasetSplit",
        base="CIFAR100",
        num_classes=100,
        train=False,
        num_images=-1,
        data_prefix="data/torchvision/cifar100",
        samples_per_gpu=128,
        workers_per_gpu=4,
        seed=seed,
        pipeline=__test_pipeline,
        download=True,
    ),
)

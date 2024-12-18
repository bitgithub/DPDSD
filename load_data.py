import torch
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizerFast


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_data_mnist(args, train_batch_size):

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        # batch_size=args.batch_size,
        batch_size=train_batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        # shuffle=True,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader


def load_data_cifar10(args, train_batch_size):
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    train_transform = transforms.Compose(
        augmentations + normalize if args.disable_dp else normalize
    )

    test_transform = transforms.Compose(normalize)

    train_dataset = CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=args.batch_size,
        batch_size=train_batch_size,
        generator=None,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_dataset = CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.workers,
    )

    return train_loader, test_loader


def load_data_fashionmnist(args, train_batch_size):

    FASHION_MNIST_MEAN = 0.286
    FASHION_MNIST_STD = 0.3529

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
                ]
            ),
        ),
        # batch_size=args.batch_size,
        batch_size=train_batch_size,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader


def load_data_imdb(args, train_batch_size):
    raw_dataset = load_dataset("imdb", cache_dir=args.data_root)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    dataset = raw_dataset.map(
        lambda x: tokenizer(
            x["text"], truncation=True, max_length=args.max_sequence_length
        ),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "label"])

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.workers,
        # batch_size=args.batch_size,
        batch_size=train_batch_size,
        collate_fn=padded_collate,
        pin_memory=True,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=padded_collate,
        pin_memory=True,
    )

    return train_loader, test_loader


def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [elem["input_ids"] for elem in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    y = torch.stack([elem["label"] for elem in batch]).long()
    return x, y

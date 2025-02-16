"""
Author: Anonymous during review process
Institution: Anonymous during review process
Date: Anonymous during review process
Load CIFAR datasets for the experiments.
"""
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split


def load_cifar_dataset(dataset_class, batch_size, seed, mask_batch):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load training dataset
    if dataset_class == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True, transform=transform_train)
        
        test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                                download=True, transform=transform_test)

    elif dataset_class == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True,
                                                download=True, transform=transform_train)
        
        test_dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False,
                                                download=True, transform=transform_test)

    labels = [label for _, label in train_dataset]
    
    train_idx, valid_idx, _, _ = train_test_split(
        range(len(train_dataset)),
        labels,
        stratify=labels,  # Stratified keeps the same distribution of classes
        test_size=0.2,
        random_state=42
    )

    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(train_dataset, valid_idx)
    
    trainloader = DataLoader(train_subset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    validloader = DataLoader(valid_subset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    testloader = DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    if mask_batch > 4096:
        mask_batch = 4096 # Max handled by the GPU Quadro RTX 6000 with 24GB of memory
        # Mask computation adjustment for extreme case is done in compressors.py
    
    maskloader = DataLoader(train_subset, batch_size=mask_batch,
                                            shuffle=True, num_workers=2)

    return trainloader, validloader, testloader, maskloader
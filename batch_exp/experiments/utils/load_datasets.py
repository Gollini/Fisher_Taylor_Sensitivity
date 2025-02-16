"""
Author: Anonymous during review process
Institution: Anonymous during review process
Date: Anonymous during review process
Load datasets for the experiments.
"""
from data.cifar import load_cifar_dataset

DATASETS = [
    "cifar10",
    "cifar100",
]

def init_dataset(params, seed, mask_batch):
        if params["class"] not in DATASETS:
            raise ValueError("""Dataset is not recognized.""")
        
        if params["class"] == "cifar10" or params["class"] == "cifar100":
            trainloader, validloader, testloader, maskloader = load_cifar_dataset(
                params["class"], params["batch_size"], seed, mask_batch
            )
        
        print(f"Dataset {params['class']} loaded")
        return trainloader, validloader, testloader, maskloader
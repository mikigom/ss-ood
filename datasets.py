from tqdm import tqdm
import cv2
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST, CIFAR10, MNIST


class FashionMNIST_Redefined(FashionMNIST):
    """
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    classes = ['Ankle boot',
               'Bag',
               'Coat',
               'Dress',
               'Pullover',
               'Sandal',
               'Shirt',
               'Sneaker',
               'T-shirt/top',
               'Trouser']
    """

    re_mapping_classes = {0: 8,
                          1: 9,
                          2: 4,
                          3: 3,
                          4: 2,
                          5: 5,
                          6: 6,
                          7: 7,
                          8: 1,
                          9: 0}

    def __init__(self, in_class: int, root: str, transform, is_training: bool):
        from tqdm import tqdm
        tqdm.monitor_interval = 0
        super(FashionMNIST_Redefined, self).__init__(root, train=is_training, transform=transform, download=True)

        self.targets = [FashionMNIST_Redefined.re_mapping_classes[int(target)] for target in self.targets]

        if is_training:
            idxs_in_class = [i for i, target in enumerate(self.targets) if target == in_class]
            self.data = [self.data[idx_in_class] for idx_in_class in idxs_in_class]
            self.targets = [self.targets[idx_in_class] for idx_in_class in idxs_in_class]
        else:
            self.targets = [int(self.targets[i] == in_class) for i in range(len(self.targets))]


class CIFAR10_Redefined(CIFAR10):
    def __init__(self, in_class: int, root: str, transform, is_training: bool):
        super(CIFAR10_Redefined, self).__init__(root, train=is_training, transform=transform, download=True)

        if is_training:
            idxs_in_class = [i for i, target in enumerate(self.targets) if target == in_class]
            self.data = [self.data[idx_in_class] for idx_in_class in idxs_in_class]
            self.targets = [self.targets[idx_in_class] for idx_in_class in idxs_in_class]
        else:
            self.targets = [int(self.targets[i] == in_class) for i in range(len(self.targets))]


class MNIST_Redefined(MNIST):
    def __init__(self, in_class: int, root: str, transform, is_training: bool):
        super(MNIST_Redefined, self).__init__(root, train=is_training, transform=transform, download=True)
        from tqdm import tqdm
        tqdm.monitor_interval = 0

        if is_training:
            idxs_in_class = [i for i, target in enumerate(self.targets) if target == in_class]
            self.data = [self.data[idx_in_class] for idx_in_class in idxs_in_class]
            self.targets = [self.targets[idx_in_class] for idx_in_class in idxs_in_class]
        else:
            self.targets = [int(self.targets[i] == in_class) for i in range(len(self.targets))]

"""
class CatDog(Dataset):
    def __init__(self, in_class,
                 catdog_path="/mnt/nas/workspace/sjh/anomaly-gpnd/dataset_utils/cat_dog/dogs-vs-cats/train"):
        self.catdog_path = catdog_path
"""

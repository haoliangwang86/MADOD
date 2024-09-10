# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import random
import torch
import torchvision.datasets.folder

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import TensorDataset, ConcatDataset
from torchvision.datasets import MNIST, ImageFolder, DatasetFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import rotate
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
from typing import Any, Callable, Dict, List, Optional, Tuple

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        original_dataset_tr = MNIST(root, train=True, download=True, transform=transform)
        original_dataset_te = MNIST(root, train=False, download=True, transform=transform)

        data = ConcatDataset([original_dataset_tr, original_dataset_te])
        original_images = torch.cat([img for img, _ in data])
        original_labels = torch.cat([torch.tensor(label).unsqueeze(0) for _, label in data])

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class MultipleEnvironmentMNISTOOD(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes, ood_classes, exclude_oods_from_training, exclude_oods_from_testing):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        original_dataset_tr = MNIST(root, train=True, download=True, transform=transform)
        original_dataset_te = MNIST(root, train=False, download=True, transform=transform)

        data = ConcatDataset([original_dataset_tr, original_dataset_te])
        original_images = torch.cat([img for img, _ in data])
        original_labels = torch.cat([torch.tensor(label).unsqueeze(0) for _, label in data])

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i], ood_classes, exclude_oods_from_training, exclude_oods_from_testing))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 32, 32,), 2)

        self.input_shape = (2, 32, 32,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):

        # Assign a binary label based on the digit
        labels = (labels < 5).float()

        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)

        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float() #.div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class ColoredMNISTOOD(MultipleEnvironmentMNISTOOD):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams, ood_classes, exclude_oods_from_training=True, exclude_oods_from_testing=True):
        self.test_envs = test_envs
        self.ood_classes = ood_classes
        self.corr_level = [0.1, 0.2, 0.9]

        super(ColoredMNISTOOD, self).__init__(root, self.corr_level,
                                         self.color_dataset, (2, 32, 32,), 2, ood_classes, exclude_oods_from_training, exclude_oods_from_testing)

        self.input_shape = (2, 32, 32,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment, ood_classes, exclude_oods_from_training, exclude_oods_from_testing):
        test_env_corrs = [self.corr_level[i] for i in self.test_envs]

        ood_mask = sum(labels == i for i in ood_classes).bool()
        ind_mask = ~ood_mask

        if environment not in test_env_corrs:  # filter out ood classes for training domains
            if exclude_oods_from_training:
                images = images[ind_mask]
                labels = labels[ind_mask]
                print(f'Class {ood_classes} filtered out for env {environment}!')
        else:  # filter out ood classes for test domain
            if exclude_oods_from_testing:
                images = images[ind_mask]
                labels = labels[ind_mask]
                print(f'Class {ood_classes} filtered out for env {environment}!')

        # Assign a binary label based on the digit
        labels = (labels < 5).float()

        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)

        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        # TODO: does not work when there are more than one OOD classes, does not work when OOD class is 0 or 1
        # revert ood class label
        if environment in test_env_corrs and not exclude_oods_from_testing:
            labels = labels.masked_fill(ood_mask == True, ood_classes[0])

        x = images.float() #.div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class ColoredMNISTMetaOOD(MultipleEnvironmentMNISTOOD):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams, ood_classes, exclude_oods_from_training=True, exclude_oods_from_testing=True, no_error=False):
        self.test_envs = test_envs
        self.ood_classes = ood_classes
        self.corr_level = [0.1, 0.2, 0.9]
        self.no_error = no_error

        super(ColoredMNISTMetaOOD, self).__init__(root, self.corr_level,
                                         self.color_dataset, (2, 32, 32,), 2, ood_classes, exclude_oods_from_training, exclude_oods_from_testing)

        self.input_shape = (2, 32, 32,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment, ood_classes, exclude_oods_from_training, exclude_oods_from_testing):
        test_env_corrs = [self.corr_level[i] for i in self.test_envs]

        ood_mask = sum(labels == i for i in ood_classes).bool()
        ind_mask = ~ood_mask

        if environment not in test_env_corrs:  # filter out ood classes for training domains
            if exclude_oods_from_training:
                images = images[ind_mask]
                labels = labels[ind_mask]
                print(f'Class {ood_classes} filtered out for env {environment}!')
        else:  # filter out ood classes for test domain
            if exclude_oods_from_testing:
                images = images[ind_mask]
                labels = labels[ind_mask]
                print(f'Class {ood_classes} filtered out for env {environment}!')

        # Assign a binary label based on the digit
        labels = (labels < 5).float()

        if not self.no_error:
            # Flip label with probability 0.25
            labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)

        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        # TODO: does not work when there are more than one OOD classes, does not work when OOD class is 0 or 1
        # revert ood class label
        if environment in test_env_corrs and not exclude_oods_from_testing:
            labels = labels.masked_fill(ood_mask == True, ood_classes[0])

        # labels = labels.masked_fill(ood_mask == True, ood_classes[0])  # only for single domain experiment

        x = images.float()  # .div_(255.0)
        y = labels.view(-1).long()

        # Create new tensor x_neg with the same size as x
        x_neg = torch.zeros_like(x)

        non_matching_imgs = {}

        # Iterate through the unique labels in y
        for label in y.unique():
            # Get the indices of the images in x that have the current label
            non_matching_indices = (y != label).nonzero(as_tuple=True)[0]
            non_matching_imgs[label.item()] = x[non_matching_indices]

        for i in range(len(x)):
            random_idx = random.randrange(len(non_matching_imgs[y[i].item()]))
            x_neg[i] = non_matching_imgs[y[i].item()][random_idx]

        return TensorDataset(x, y, x_neg)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


# # for backbone experiments
# class MultipleEnvironmentImageFolder(MultipleDomainDataset):
#     def __init__(self, root, test_envs, augment, hparams):
#         super().__init__()
#         environments = [f.name for f in os.scandir(root) if f.is_dir()]
#         environments = sorted(environments)
#
#         transform = transforms.Compose([
#             transforms.Resize((112,112)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#         augment_transform = transforms.Compose([
#             # transforms.Resize((112,112)),
#             transforms.RandomResizedCrop(112, scale=(0.7, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#             transforms.RandomGrayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#
#         self.datasets = []
#         for i, environment in enumerate(environments):
#
#             if augment and (i not in test_envs):
#                 env_transform = augment_transform
#             else:
#                 env_transform = transform
#
#             path = os.path.join(root, environment)
#             env_dataset = ImageFolder(path,
#                 transform=env_transform)
#
#             self.datasets.append(env_dataset)
#
#         self.input_shape = (3, 112, 112,)
#         self.num_classes = len(self.datasets[-1].classes)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
class ImageFolderOOD(DatasetFolder):
    def __init__(
            self,
            root: str,
            ood_classes: List,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        self.ood_classes = ood_classes
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        ood_classes = sorted(self.ood_classes, reverse=True)
        for ood_class in ood_classes:
            del classes[ood_class]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class DatasetFolderMeta(DatasetFolder):
    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        n_pseudo: int = 1,
        k_support: int = 5,  # Support set shots
        k_query: int = 5,  # Query set shots
        k_pseudo: int = 5,  # Pseudo-OOD shots
        use_relative_labels: bool = True,  # Control relative labeling
    ):
        super().__init__(root, loader, extensions, transform, target_transform, is_valid_file)
        self.pseudo_ood_samples = None
        self.ind_samples = None
        self.ind_classes = None
        self.pseudo_ood_classes = None
        self.n_pseudo = n_pseudo
        self.k_support = k_support
        self.k_query = k_query
        self.k_pseudo = k_pseudo
        self.use_relative_labels = use_relative_labels
        self.prepare_data()

    def prepare_data(self):
        all_classes = list(self.class_to_idx.keys())
        if self.n_pseudo > 0:
            self.pseudo_ood_classes = random.sample(all_classes, self.n_pseudo)
            self.ind_classes = [cls for cls in all_classes if cls not in self.pseudo_ood_classes]
            self.pseudo_ood_samples = {cls: [] for cls in self.pseudo_ood_classes}
        else:
            self.pseudo_ood_classes = []
            self.ind_classes = all_classes
            self.pseudo_ood_samples = {}

        self.ind_samples = {cls: [] for cls in self.ind_classes}

        for path, label in self.samples:
            class_name = self.classes[label]
            if class_name in self.ind_classes:
                self.ind_samples[class_name].append((path, label))
            elif class_name in self.pseudo_ood_classes:
                self.pseudo_ood_samples[class_name].append((path, label))

    def __getitem__(self, index: int) -> Tuple[List[Any], List[int], List[Any], List[int], List[Any]]:
        support_set = []
        support_labels = []
        query_set = []
        query_labels = []

        # Randomly select classes for this episode
        episode_classes = random.sample(self.ind_classes, len(self.ind_classes))

        for i, cls in enumerate(episode_classes):
            # Get available samples for this class
            available_samples = self.ind_samples[cls]
            required_samples = self.k_support + self.k_query

            # If we don't have enough samples, we'll need to duplicate some
            if len(available_samples) < required_samples:
                # First, use all available samples
                samples = available_samples.copy()

                # Then, randomly duplicate samples until we have enough
                while len(samples) < required_samples:
                    samples.append(random.choice(available_samples))
            else:
                # If we have enough samples, just randomly sample as before
                samples = random.sample(available_samples, required_samples)

            random.shuffle(samples)

            # First k_support samples for support set
            for path, label in samples[:self.k_support]:
                img = self.loader(path)
                if self.transform:
                    img = self.transform(img)
                support_set.append(img)
                support_labels.append(i if self.use_relative_labels else label)  # true or relative label

            # Remaining k_query samples for query set
            if self.k_query > 0:
                for path, label in samples[self.k_support:]:
                    img = self.loader(path)
                    if self.transform:
                        img = self.transform(img)
                    query_set.append(img)
                    query_labels.append(i if self.use_relative_labels else label)  # true or relative label

        # Pseudo-OOD set
        pseudo_ood_set = []
        if self.n_pseudo > 0:
            for cls in self.pseudo_ood_classes:
                available_samples = self.pseudo_ood_samples[cls]
                if len(available_samples) < self.k_pseudo:
                    # creating a list of samples that is at least as long as self.k_pseudo by repeating the available_samples list
                    samples = available_samples * (self.k_pseudo // len(available_samples) + 1)
                    samples = samples[:self.k_pseudo]
                else:
                    samples = random.sample(available_samples, self.k_pseudo)

                for path, _ in samples:
                    img = self.loader(path)
                    if self.transform:
                        img = self.transform(img)
                    pseudo_ood_set.append(img)

        # Shuffle the sets
        combined = list(zip(support_set, support_labels))
        random.shuffle(combined)
        support_set, support_labels = zip(*combined)

        if self.k_query > 0:
            combined = list(zip(query_set, query_labels))
            random.shuffle(combined)
            query_set, query_labels = zip(*combined)

        if self.n_pseudo > 0:
            random.shuffle(pseudo_ood_set)

        # Convert to torch tensors
        support_set = torch.stack([torch.as_tensor(img) for img in support_set])
        support_labels = torch.tensor(support_labels, dtype=torch.long)

        if query_set:
            query_set = torch.stack([torch.as_tensor(img) for img in query_set])
            query_labels = torch.tensor(query_labels, dtype=torch.long)
        else:
            query_set = torch.empty(0)  # Empty tensor if no query samples
            query_labels = torch.empty(0, dtype=torch.long)  # Empty tensor if no query samples

        if pseudo_ood_set:
            pseudo_ood_set = torch.stack([torch.as_tensor(img) for img in pseudo_ood_set])
        else:
            pseudo_ood_set = torch.empty(0)  # Empty tensor if no pseudo-OOD samples

        return support_set, support_labels, query_set, query_labels, pseudo_ood_set

    def __len__(self) -> int:
        return 10000000  # A large number to simulate "infinite" batches


class ImageFolderMeta(DatasetFolderMeta):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            n_pseudo: int = 1,
            k_support: int = 5,
            k_query: int = 5,
            k_pseudo: int = 5,
            use_relative_labels: bool = True,  # Control relative labeling
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            n_pseudo=n_pseudo,
            k_support=k_support,
            k_query=k_query,
            k_pseudo=k_pseudo,
            use_relative_labels=use_relative_labels,
        )
        self.imgs = self.samples


class ImageFolderMetaOOD(DatasetFolderMeta):
    def __init__(
            self,
            root: str,
            ood_classes: List,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            n_pseudo: int = 1,
            k_support: int = 5,
            k_query: int = 5,
            k_pseudo: int = 5,
            use_relative_labels: bool = True,  # Control relative labeling
    ):
        self.ood_classes = ood_classes
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            n_pseudo=n_pseudo,
            k_support=k_support,
            k_query=k_query,
            k_pseudo=k_pseudo,
            use_relative_labels=use_relative_labels,
        )
        self.imgs = self.samples

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        ood_classes = sorted(self.ood_classes, reverse=True)
        for ood_class in ood_classes:
            del classes[ood_class]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class MultipleEnvironmentImageFolderOOD(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams, ood_classes, no_data_aug, exclude_oods_from_training, exclude_oods_from_testing):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            # if doing evaluation, don't use data aug
            if no_data_aug:
                env_transform = transform
            else:
                if augment and (i not in test_envs):
                    env_transform = augment_transform
                else:
                    env_transform = transform

            path = os.path.join(root, environment)

            # exclude ood samples
            if i in test_envs:
                if exclude_oods_from_testing:
                    env_dataset = ImageFolderOOD(path,
                                                 transform=env_transform, ood_classes=ood_classes)
                else:
                    env_dataset = ImageFolder(path,
                                              transform=env_transform)
            else:
                if exclude_oods_from_training:
                    env_dataset = ImageFolderOOD(path,
                                                 transform=env_transform, ood_classes=ood_classes)
                else:
                    env_dataset = ImageFolder(path,
                                              transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)

        # only count the classes for classification
        if exclude_oods_from_testing:
            self.num_classes = len(self.datasets[test_envs[0]].classes)
        else:
            self.num_classes = len(self.datasets[test_envs[0]].classes) - 1


# # for backbone experiments
# class MultipleEnvironmentImageFolderOOD(MultipleDomainDataset):
#     def __init__(self, root, test_envs, augment, hparams, ood_classes, no_data_aug, exclude_oods_from_training, exclude_oods_from_testing):
#         super().__init__()
#         environments = [f.name for f in os.scandir(root) if f.is_dir()]
#         environments = sorted(environments)
#
#         transform = transforms.Compose([
#             transforms.Resize((112, 112)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#         augment_transform = transforms.Compose([
#             # transforms.Resize((112,112)),
#             transforms.RandomResizedCrop(112, scale=(0.7, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#             transforms.RandomGrayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#
#         self.datasets = []
#         for i, environment in enumerate(environments):
#
#             # if doing evaluation, don't use data aug
#             if no_data_aug:
#                 env_transform = transform
#             else:
#                 if augment and (i not in test_envs):
#                     env_transform = augment_transform
#                 else:
#                     env_transform = transform
#
#             path = os.path.join(root, environment)
#
#             # exclude ood samples
#             if i in test_envs:
#                 if exclude_oods_from_testing:
#                     env_dataset = ImageFolderOOD(path,
#                                                  transform=env_transform, ood_classes=ood_classes)
#                 else:
#                     env_dataset = ImageFolder(path,
#                                               transform=env_transform)
#             else:
#                 if exclude_oods_from_training:
#                     env_dataset = ImageFolderOOD(path,
#                                                  transform=env_transform, ood_classes=ood_classes)
#                 else:
#                     env_dataset = ImageFolder(path,
#                                               transform=env_transform)
#
#             self.datasets.append(env_dataset)
#
#         self.input_shape = (3, 112, 112,)
#
#         # only count the classes for classification
#         if exclude_oods_from_testing:
#             self.num_classes = len(self.datasets[test_envs[0]].classes)
#         else:
#             self.num_classes = len(self.datasets[test_envs[0]].classes) - 1


class MultipleEnvironmentImageFolderMetaOOD(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams, ood_classes,
                 no_data_aug, exclude_oods_from_training, exclude_oods_from_testing,
                 n_pseudo, k_support, k_query, k_pseudo, use_relative_labels):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            # if doing evaluation, don't use data aug
            if no_data_aug:
                env_transform = transform
            else:
                if augment and (i not in test_envs):
                    env_transform = augment_transform
                else:
                    env_transform = transform

            path = os.path.join(root, environment)

            # exclude ood samples
            if i in test_envs:
                if exclude_oods_from_testing:
                    env_dataset = ImageFolderMetaOOD(path,
                                                     ood_classes=ood_classes,
                                                     transform=env_transform,
                                                     n_pseudo=n_pseudo,
                                                     k_support=k_support, k_query=k_query, k_pseudo=k_pseudo,
                                                     use_relative_labels=use_relative_labels)
                else:
                    env_dataset = ImageFolderMeta(path,
                                                  transform=env_transform,
                                                  n_pseudo=n_pseudo,
                                                  k_support=k_support, k_query=k_query, k_pseudo=k_pseudo,
                                                  use_relative_labels=use_relative_labels)
            else:
                if exclude_oods_from_training:
                    env_dataset = ImageFolderMetaOOD(path,
                                                     ood_classes=ood_classes,
                                                     transform=env_transform,
                                                     n_pseudo=n_pseudo,
                                                     k_support=k_support, k_query=k_query, k_pseudo=k_pseudo,
                                                     use_relative_labels=use_relative_labels)
                else:
                    env_dataset = ImageFolderMeta(path,
                                                  transform=env_transform,
                                                  n_pseudo=n_pseudo,
                                                  k_support=k_support, k_query=k_query, k_pseudo=k_pseudo,
                                                  use_relative_labels=use_relative_labels)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)

        # only count the classes for classification
        if exclude_oods_from_testing:
            self.num_classes = len(self.datasets[test_envs[0]].classes)
        else:
            self.num_classes = len(self.datasets[test_envs[0]].classes) - 1


# # for backbone experiments
# class MultipleEnvironmentImageFolderMetaOOD(MultipleDomainDataset):
#     def __init__(self, root, test_envs, augment, hparams, ood_classes, no_data_aug, exclude_oods_from_training, exclude_oods_from_testing):
#         super().__init__()
#         environments = [f.name for f in os.scandir(root) if f.is_dir()]
#         environments = sorted(environments)
#
#         transform = transforms.Compose([
#             transforms.Resize((112, 112)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#         augment_transform = transforms.Compose([
#             # transforms.Resize((112,112)),
#             transforms.RandomResizedCrop(112, scale=(0.7, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#             transforms.RandomGrayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#
#         self.datasets = []
#         for i, environment in enumerate(environments):
#
#             # if doing evaluation, don't use data aug
#             if no_data_aug:
#                 env_transform = transform
#             else:
#                 if augment and (i not in test_envs):
#                     env_transform = augment_transform
#                 else:
#                     env_transform = transform
#
#             path = os.path.join(root, environment)
#
#             # exclude ood samples
#             if i in test_envs:
#                 if exclude_oods_from_testing:
#                     env_dataset = ImageFolderMetaOOD(path,
#                                                  transform=env_transform, ood_classes=ood_classes)
#                 else:
#                     env_dataset = ImageFolderMeta(path,
#                                               transform=env_transform)
#             else:
#                 if exclude_oods_from_training:
#                     env_dataset = ImageFolderMetaOOD(path,
#                                                  transform=env_transform, ood_classes=ood_classes)
#                 else:
#                     env_dataset = ImageFolderMeta(path,
#                                               transform=env_transform)
#
#             self.datasets.append(env_dataset)
#
#         self.input_shape = (3, 112, 112,)
#
#         # only count the classes for classification
#         if exclude_oods_from_testing:
#             self.num_classes = len(self.datasets[test_envs[0]].classes)
#         else:
#             self.num_classes = len(self.datasets[test_envs[0]].classes) - 1

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "TerraIncognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class PACSOOD(MultipleEnvironmentImageFolderOOD):
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams, ood_classes,
                 no_data_aug=False, exclude_oods_from_training=True, exclude_oods_from_testing=True):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams, ood_classes,
                         no_data_aug, exclude_oods_from_training, exclude_oods_from_testing)


class VLCSOOD(MultipleEnvironmentImageFolderOOD):
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams, ood_classes,
                 no_data_aug=False, exclude_oods_from_training=True, exclude_oods_from_testing=True):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams, ood_classes,
                         no_data_aug, exclude_oods_from_training, exclude_oods_from_testing)


class TerraIncognitaOOD(MultipleEnvironmentImageFolderOOD):
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams, ood_classes,
                 no_data_aug=False, exclude_oods_from_training=True, exclude_oods_from_testing=True):
        self.dir = os.path.join(root, "TerraIncognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams, ood_classes,
                         no_data_aug, exclude_oods_from_training, exclude_oods_from_testing)


class PACSMetaOOD(MultipleEnvironmentImageFolderMetaOOD):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams, ood_classes,
                 no_data_aug=False, exclude_oods_from_training=True, exclude_oods_from_testing=True,
                 n_pseudo=1, k_support=5, k_query=5, k_pseudo=5, use_relative_labels=True):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams, ood_classes,
                         no_data_aug, exclude_oods_from_training, exclude_oods_from_testing,
                         n_pseudo, k_support, k_query, k_pseudo, use_relative_labels)


class VLCSMetaOOD(MultipleEnvironmentImageFolderMetaOOD):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams, ood_classes,
                 no_data_aug=False, exclude_oods_from_training=True, exclude_oods_from_testing=True,
                 n_pseudo=1, k_support=5, k_query=5, k_pseudo=5, use_relative_labels=True):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams, ood_classes,
                         no_data_aug, exclude_oods_from_training, exclude_oods_from_testing,
                         n_pseudo, k_support, k_query, k_pseudo, use_relative_labels)


class TerraIncognitaMetaOOD(MultipleEnvironmentImageFolderMetaOOD):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams, ood_classes,
                 no_data_aug=False, exclude_oods_from_training=True, exclude_oods_from_testing=True,
                 n_pseudo=1, k_support=5, k_query=5, k_pseudo=5, use_relative_labels=True):
        self.dir = os.path.join(root, "TerraIncognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams, ood_classes,
                         no_data_aug, exclude_oods_from_training, exclude_oods_from_testing,
                         n_pseudo, k_support, k_query, k_pseudo, use_relative_labels)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 500
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 96, 96)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomResizedCrop(96, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)

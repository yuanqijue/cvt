import os
import moco.builder
import moco.loader
import moco.optimizer

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader


def build_transform(is_train, args, aug=True):
    if args.data_set == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif args.data_set == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    elif args.data_set == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if not aug:
        return transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor(), normalize, ])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [transforms.RandomResizedCrop(args.image_size, scale=(args.crop_min, 1.)),
                     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                                             ], p=0.8), transforms.RandomGrayscale(p=0.2),
                     transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
                     transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]

    augmentation2 = [transforms.RandomResizedCrop(args.image_size, scale=(args.crop_min, 1.)),
                     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                                             ], p=0.8), transforms.RandomGrayscale(p=0.2),
                     transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
                     transforms.RandomApply([moco.loader.Solarize()], p=0.2), transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), normalize]

    transform_train = moco.loader.TwoCropsTransform(transforms.Compose(augmentation1),
                                                    transforms.Compose(augmentation2))
    return transform_train


def build_dataset(is_train, args, aug=True):
    transform = build_transform(is_train, args, aug)

    if args.data_set == 'imagenet':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'cifar10':
        dataset = datasets.CIFAR10(root=args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.data_set == "cifar100":
        dataset = datasets.CIFAR100(root=args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif args.data_set == "svhn":
        dataset = datasets.SVHN(root=args.data_path, split='train' if is_train else 'test', download=True,
                                transform=transform)
        nb_classes = 10
    elif args.data_set == "lsun":
        dataset = datasets.LSUN(root=args.data_path, classes='train' if is_train else 'test', transform=transform)
        nb_classes = 20
    elif args.data_set == "place365":
        dataset = datasets.Places365(root=args.data_path, split='train-standard' if is_train else 'val', download=True,
                                     small=True, transform=transform)
        nb_classes = 365
    else:
        raise NotImplementedError("Only [imagenet, cifar10, cifar100, svhn, lsun, place365 ] are supported")

    return dataset, nb_classes

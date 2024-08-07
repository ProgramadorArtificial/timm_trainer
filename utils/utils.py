import torch
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .custom_dataloader import CustomDataset


# Font: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
def calculate_mean_std(folder_path, image_size, batch_size=128):
    """
    Calculate the mean and standard of all images in a folder to use to normalize the data

    Args:
        folder_path: Path to the folder containing the images to calculate mean and standard values
        image_size: Size of the images used for training
        batch_size: Batch size for data loader to calculate mean and standard values
    Returns:
        (list, list): Mean and std of all images
    """
    print(f'Calculating mean and std of all images in {folder_path}')
    train_dataset = CustomDataset(folder_path, transform=default_transforms([], [], image_size),
                                  load_images_memory=False)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    nb_samples = 0
    for imgs, _ in tqdm(loader):
        input = imgs
        # [batch_size x 3 x image_size x image_size]
        psum += input.sum(axis=[0, 2, 3])
        psum_sq += (input ** 2).sum(axis=[0, 2, 3])
        nb_samples += input.size(0)

    # pixel count
    count = nb_samples * image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    return total_mean.numpy().tolist(), total_std.numpy().tolist()


def train_transforms(mean, std, image_size):
    """
    Transforms with augmentation, used for training

    Args:
        mean (list): Normalize mean value
        std (list): Normalize standard value
        image_size (int): Size of the images used for training
    """
    if mean and std:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.3),
            T.RandomApply(torch.nn.ModuleList(
                [T.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.6), hue=(-0.05, 0.05)),
                 T.GaussianBlur(3, 7),
                 T.RandomRotation(4),
                 T.RandomAdjustSharpness(3),
                 ]), p=0.4),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.3),
            T.RandomApply(torch.nn.ModuleList(
                [T.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.6), hue=(-0.05, 0.05)),
                 T.GaussianBlur(3, 7),
                 T.RandomRotation(4),
                 T.RandomAdjustSharpness(3),
                 ]), p=0.4),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])


def default_transforms(mean, std, image_size):
    """
    Default transforms, used for inferences

    Args:
        mean (list): Normalize mean value
        std (list): Normalize standard value
        image_size (int): Size of the images used for training
    """
    if mean and std:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

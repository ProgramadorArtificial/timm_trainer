import copy
import json
import pathlib
import logging
from datetime import datetime

import torch


class SaveCheckpoint:
    def __init__(self, save_path, model_name, resume_checkpoint, mean, std, image_size, loss_function, is_float16, id_to_label={}):
        """
        Prepare information to save checkpoint and start logging

        Args:
            save_path (str): path to save checkpoint and logs
            model_name (str): model name
            resume_checkpoint (str): path to checkpoint file
            mean (list): mean used to normalize images
            std (list): std used to normalize images
            image_size (int): Input image size used for training
            loss_function (std): loss function used to train the model
            is_float16 (bool): Whether to save model as float16
            id_to_label (dict): Ids to labels used for training
        """
        self.save_path = save_path
        self.folder_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_name = model_name
        self.resume_checkpoint = resume_checkpoint
        self.mean = mean
        self.std = std
        self.image_size = image_size
        self.loss_function = loss_function
        self.is_float16 = is_float16
        self.id_to_label = id_to_label

        pathlib.Path(f'{self.save_path}/{self.folder_name}').mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=f'{self.save_path}/{self.folder_name}/training.log',
            filemode='w'
        )

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

    def save(self, model, epoch, optimizer, train_epoch_loss, val_epoch_accuracy=None, val_epoch_loss=None):
        """
        Save json with checkpoint information and save checkpoint

        Args:
            model (torch.nn.Module): model to save
            epoch (int): current epoch
            optimizer (torch.optim.Optimizer): optimizer used for training
            train_epoch_loss (float): train epoch loss
            val_epoch_accuracy (float): val epoch accuracy. None if not running validation dataset
            val_epoch_loss (float): val epoch loss. None if not running validation dataset
        """
        best_results = {
            'model_name': self.model_name,
            'resume': self.resume_checkpoint,
            'best_epoch': epoch,
            'data': datetime.today().strftime('%Y/%m/%d-%H:%M'),
            'loss_function': self.loss_function,
            'optimizer': optimizer.__class__.__name__,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'mean': self.mean,
            'std': self.std,
            'image_size': self.image_size,
            'train_best_loss': train_epoch_loss,
            'val_best_accuracy': val_epoch_accuracy,
            'val_best_loss': val_epoch_loss,
        }
        torch.save({
            'model': copy.deepcopy(model).half() if self.is_float16 else model,
            'mean': self.mean,
            'std': self.std,
            'image_size': self.image_size,
            'id_to_label': self.id_to_label
        }, f'{self.save_path}/{self.folder_name}/best_{epoch}.pth')
        with open(f'{self.save_path}/{self.folder_name}/best_{epoch}_results.json', 'w') as file:
            json.dump(best_results, file, indent=4)

        return best_results

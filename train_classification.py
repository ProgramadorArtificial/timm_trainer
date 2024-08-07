"""
Script used to train classification models
"""
import time
import json
import random
import logging
from tqdm import tqdm

import timm
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from utils.save_checkpoint import SaveCheckpoint
from utils.custom_dataloader import CustomDataset
from utils.utils import calculate_mean_std, train_transforms, default_transforms

# ### Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Timm model name
MODEL_NAME = 'convnextv2_nano'  # convnextv2_nano == convnextv2_nano.fcmae_ft_in22k_in1k
IS_FLOAT16 = True

TRAIN_FOLDER = 'dataset/100_sports_image_classification/train'
# Empty string if you will not do validation
VAL_FOLDER = 'dataset/100_sports_image_classification/valid'

# ### Resume checkpoint ###
# If empty string will download model from Timm. If path to a checkpoint will use it to train
RESUME_CHECKPOINT = ''
# Whether to use all hyperparameters depending on where the checkpoint stopped (epoch, learning rate, best_loss, ...)
CONTINUE_FROM_CHECKPOINT = False  # RESUME_CHECKPOINT must be different from empty string

# Whether load all images in memory before start training or load in disk every epoch
LOAD_IMAGES_MEMORY = False
# If False will only train and consider the best checkpoint with the least training loss
IS_VALID = True if VAL_FOLDER else False

SAVE_PATH = 'checkpoints'
NUM_EPOCHS = 20000
PATIENCE = 20
BATCH_SIZE = 512
LR = 0.01
NUM_WORKERS = 8
IMAGE_SIZE = 112

LOSS_FUNCTION = 'cross_entropy'  # cross_entropy
OPTIMIZER = 'sgd'  # adam adamW sgd

# ### Scheduler ###
SCHEDULER = 'stepLR'  # '' plateau stepLR
SCHEDULER_PATIENCE = 4
SCHEDULER_GAMMA = 0.95

# Whether calculate custom MEAN and STD for the dataset or use the default
IS_CALCULATE_MEAN_STD = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_results = {
    'best_epoch': 0,
    'train_best_loss': np.inf,
    'val_best_accuracy': 0.0,
    'val_best_loss': np.inf
}
start_epoch = 1

if RESUME_CHECKPOINT:
    checkpoint = torch.load(RESUME_CHECKPOINT)
    # In case the model was saved as float16 (half), convert it to float32
    model = checkpoint['model'].to(torch.float)

    results_json = json.load(open(RESUME_CHECKPOINT.replace('.pth', '_results.json')))
    MODEL_NAME = results_json['model_name']

    if CONTINUE_FROM_CHECKPOINT:
        LR = results_json['learning_rate']
        LOSS_FUNCTION = results_json['loss_function']
        MEAN = results_json['mean']
        STD = results_json['std']
        IMAGE_SIZE = results_json['image_size']
        start_epoch = results_json['best_epoch'] + 1
        NUM_EPOCHS += start_epoch
        best_results['best_epoch'] = start_epoch
        best_results['train_best_loss'] = results_json['train_best_loss']
        best_results['val_best_accuracy'] = results_json['val_best_accuracy'] if results_json['val_best_accuracy'] else 0.0
        best_results['val_best_loss'] = results_json['val_best_loss'] if results_json['val_best_loss'] else np.inf

if not RESUME_CHECKPOINT or (RESUME_CHECKPOINT and CONTINUE_FROM_CHECKPOINT is False):
    if IS_CALCULATE_MEAN_STD:
        MEAN, STD = calculate_mean_std(TRAIN_FOLDER, IMAGE_SIZE)
    else:
        # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        MEAN, STD = timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD

train_dataset = CustomDataset(TRAIN_FOLDER, transform=train_transforms(MEAN, STD, IMAGE_SIZE),
                              return_labels_onehot=True, load_images_memory=LOAD_IMAGES_MEMORY)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

if IS_VALID:
    val_dataset = CustomDataset(VAL_FOLDER, transform=default_transforms(MEAN, STD, IMAGE_SIZE),
                                return_labels_onehot=True, load_images_memory=LOAD_IMAGES_MEMORY, onehot=train_dataset.get_onehot())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

id_to_label = train_dataset.get_onehot_classes()

# Prepare files and folders to save logging and checkpoints
save_checkpoint = SaveCheckpoint(SAVE_PATH, MODEL_NAME, RESUME_CHECKPOINT, MEAN, STD, IMAGE_SIZE, LOSS_FUNCTION,
                                 IS_FLOAT16, id_to_label)

if LOSS_FUNCTION == 'cross_entropy':
    criterion = torch.nn.CrossEntropyLoss()
else:
    raise ValueError(f'Invalid LOSS_FUNCTION ({LOSS_FUNCTION})')

if not RESUME_CHECKPOINT:
    model = timm.create_model(MODEL_NAME, pretrained=True)
    in_feat = model.head.fc.in_features
    model.head.fc = torch.nn.Linear(in_feat, train_dataset.get_num_classes())
if RESUME_CHECKPOINT:
    if LOSS_FUNCTION in ['cross_entropy'] and model.head.fc.out_features != train_dataset.get_num_classes():
        logging.warning(f'Detected checkpoint out_features different than num classes.'
                        f' Changing out_features to {train_dataset.get_num_classes()}')
        in_feat = model.head.fc.in_features
        model.head.fc = torch.nn.Linear(in_feat, train_dataset.get_num_classes())

if OPTIMIZER == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=LR)
elif OPTIMIZER == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif OPTIMIZER == 'adamW':
    optimizer = optim.AdamW(model.parameters(), lr=LR)
else:
    raise ValueError(f'Invalid OPTIMIZER ({OPTIMIZER})')

if SCHEDULER == '':
    scheduler = None
elif SCHEDULER == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=SCHEDULER_GAMMA, patience=SCHEDULER_PATIENCE,
        mode='max' if IS_VALID else 'min'  # If is_valid look at the val accuracy otherwise train loss
    )
elif SCHEDULER == 'stepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=SCHEDULER_GAMMA, step_size=SCHEDULER_PATIENCE)
else:
    raise ValueError(f'Invalid SCHEDULER ({SCHEDULER})')

# For float16 (half)
scaler = None
if IS_FLOAT16:
    scaler = GradScaler()

model.to(device)

start_training_time = time.time()
no_improvement_count = 0
try:
    for epoch in range(start_epoch, NUM_EPOCHS):
        tic = time.time()
        model.train()
        train_running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f'Train: Epoch [{epoch}/{NUM_EPOCHS}]', leave=False)
        for imgs, labels in train_loader_tqdm:
            if IS_FLOAT16:
                with autocast():
                    outputs = model(imgs.to(device))
                    optimizer.zero_grad()
                    loss = criterion(outputs, labels.to(device))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs.to(device))
                optimizer.zero_grad()
                loss = criterion(outputs, labels.to(device))

                loss.backward()
                optimizer.step()

            train_running_loss += loss.item()
            train_loader_tqdm.set_postfix({'batch_loss': loss.item()})
        train_loader_tqdm.close()
        train_epoch_loss = train_running_loss / len(train_loader)

        if not IS_VALID and train_epoch_loss < best_results['train_best_loss']:
            logging.info(f'##### New best train loss - before: {best_results['train_best_loss']} #####')
            no_improvement_count = 0
            best_results = save_checkpoint.save(model, epoch, optimizer, train_epoch_loss)

        logging.info(f'Epoch [{epoch}/{NUM_EPOCHS}] (time: {int(time.time() - tic)}), Train Loss: {train_epoch_loss:.4f}, Lr: {optimizer.param_groups[0]['lr']}')

        if IS_VALID:
            model.eval()
            val_running_loss = 0.0
            correct = 0
            total = 0
            val_loader_tqdm = tqdm(val_loader, desc=f'Val: Epoch [{epoch + 1}/{NUM_EPOCHS}]', leave=False)
            for imgs, labels in val_loader_tqdm:
                if IS_FLOAT16:
                    with autocast():
                        with torch.inference_mode():
                            outputs = model(imgs.to(device))
                            loss = criterion(outputs, labels.to(device))
                else:
                    with torch.inference_mode():
                        outputs = model(imgs.to(device))
                        loss = criterion(outputs, labels.to(device))

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                _, real = torch.max(labels.to(device), 1)
                correct += (predicted == real).sum().item()
                val_loader_tqdm.set_postfix({'batch_loss': loss.item()})

            val_loader_tqdm.close()
            val_epoch_loss = val_running_loss / len(val_loader)

            val_epoch_accuracy = correct / total
            if val_epoch_accuracy > best_results['val_best_accuracy']:
                logging.info(f'##### New best accuracy - before: {best_results['val_best_accuracy']} #####')
                no_improvement_count = 0
                best_results = save_checkpoint.save(model, epoch, optimizer, train_epoch_loss, val_epoch_accuracy, val_epoch_loss)
            logging.info(f'Epoch [{epoch}/{NUM_EPOCHS}], Val Acc: {val_epoch_accuracy:.4f}, Val Loss: {val_epoch_loss:.4f}')

        if no_improvement_count >= PATIENCE:
            logging.info(f'Early stopping: No improvement for {PATIENCE} consecutive epochs')
            break
        no_improvement_count += 1

        # Reduce learning rate
        if SCHEDULER == 'plateau':
            if IS_VALID:
                scheduler.step(val_epoch_accuracy)
            else:
                scheduler.step(train_epoch_loss)
        elif SCHEDULER == 'stepLR':
            scheduler.step()

except KeyboardInterrupt:
    logging.warning('Keyboard Interrupt: Stopping training')

logging.info(f'Training completed in {int((time.time() - start_training_time) / 60)} minutes - '
             f'Best Train Loss: {best_results["train_best_loss"]:.4f} - '
             f'Best Val Accuracy: {best_results["val_best_accuracy"]} - '
             f'Best Val Loss: {best_results["val_best_loss"]}')

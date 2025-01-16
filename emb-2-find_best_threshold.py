import os
import copy
import json
import pathlib

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils.utils import default_transforms


def f1_score(TP, FP, TN, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    if precision + recall == 0:
        return 0.0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


PATH_MODEL = 'checkpoints/20250116-120128/best_45.pth'
TEST_FOLDER = 'dataset/face-recognition-dataset/test'

IS_FLOAT16 = True

# Whether to save the best threshold inside checkpoint (best F1 score)
is_save_best_threshold = True

# How many negative comparisons for each positive comparison
NEGATIVE_MULTIPLIER = 1  # 1 == balanced comparison, 2 == double of negatives

# Test N linearly separated thresholds between the minimum and maximum distance
NUM_THRESHOLDS_TO_TEST = 2000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(PATH_MODEL)
model = copy.deepcopy(checkpoint['model'])
if IS_FLOAT16:
    model.to(device).half()
else:
    model.to(device).float()
model.eval()

transforms = default_transforms(
    mean=checkpoint['mean'],
    std=checkpoint['std'],
    image_size=checkpoint['image_size'],
)

outputs = {}
def hook(module, input, output):
    outputs['embedding'] = output
# Hook - Used to get embeddings from intermediate layers
# To use Hook set "is_hook" to True and configure the layers that will be used
is_hook = False
if is_hook:
    # Layer to use (hook)
    hook_handle = model.head.pre_logits.register_forward_hook(hook)

# ### Get embedding of all images
embs_fc = {}
embs_hook = {}
print('Getting embeddings...')
for folder in tqdm(os.listdir(TEST_FOLDER)):
    for filename in os.listdir(f'{TEST_FOLDER}/{folder}'):
        img = transforms(Image.fromarray(cv2.imread(f'{TEST_FOLDER}/{folder}/{filename}')))
        result_dists = []
        if folder not in embs_fc:
            embs_fc[folder] = []
            embs_hook[folder] = []

        # Without hook
        embs_fc[folder].append([filename, model(img.unsqueeze(0).to(device).half()).cpu().detach()])
        # With hook
        if is_hook:
            embs_hook[folder].append([filename, outputs['embedding'].cpu().detach()])

# Total negatives for each image
print('Getting negative combinations...')
negative_combinations_to_use = []
for anchor_label, embs in tqdm(embs_fc.items()):
    anchor_idx = 0
    for anchor_filename, emb in embs:
        negative_classes = list(embs_fc.keys()).copy()
        negative_classes.remove(anchor_label)
        for count in range((len(embs)-anchor_idx-1) * NEGATIVE_MULTIPLIER):
            # Get different class
            negative_label = np.random.choice(negative_classes)
            idx = np.random.choice(range(len(embs_fc[negative_label])))
            negative_combinations_to_use.append([anchor_label, anchor_filename, anchor_idx, negative_label, embs_fc[negative_label][idx][0], idx])
            #negative_classes.remove(negative_label)
        anchor_idx += 1

best_f1 = 0.0
best_threshold = 0.0

# ### Compare image by image
for layer, embs in zip(['fc', 'hook'], [embs_fc, embs_hook]):
    if layer == 'hook' and not is_hook:
        continue
    print(f'Comparing embeddings and processing threshold for {layer}...')
    results = []
    dists = {'true': [], 'false': []}
    history = {'img1': [], 'img2': [], 'dist': [], 'label': []}
    print('Comparing positive combinations...')
    for folder1, embs1 in tqdm(embs.items()):
        for idx1, values1 in enumerate(embs1):
            emb1 = values1[1]
            is_continue2 = True
            for idx2, values2 in enumerate(embs1):
                #print(values2)
                emb2 = values2[1]
                # Not compare same idxs more than one time
                if is_continue2:
                    if idx1 == idx2:
                        is_continue2 = False
                        continue
                    else:
                        continue
                # cosine_similarity for cosine
                dists['true'].append(round(float(torch.nn.functional.pairwise_distance(emb1, emb2).cpu()), 2))
                history['dist'].append(dists['true'][-1])
                history['label'].append('true')
                history['img1'].append(f'{folder1}/{values1[0]}')
                history['img2'].append(f'{folder1}/{values2[0]}')
    
    print('Comparing negative combinations...')
    for folder1, filename1, idx1, folder2, filename2, idx2 in negative_combinations_to_use:
        emb1 = embs_fc[folder1][idx1][1]
        emb2 = embs_fc[folder2][idx2][1]
        # cosine_similarity for cosine
        dists['false'].append(round(float(torch.nn.functional.pairwise_distance(emb1, emb2).cpu()), 2))
        history['dist'].append(dists['false'][-1])
        history['label'].append('false')
        history['img1'].append(f'{folder1}/{filename1}')
        history['img2'].append(f'{folder2}/{filename2}')
    
    pathlib.Path('results').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(f'results/comparations_{layer}.csv', index=False)

    # ### Find best threshold
    df_true = pd.DataFrame(dists['true'])
    df_false = pd.DataFrame(dists['false'])
    thresholds = np.linspace(min(dists['true']), max(dists['true']), NUM_THRESHOLDS_TO_TEST)
    rights = [[len(df_true[df_true[0] <= t]), len(df_false[df_false[0] > t]), t] for t in thresholds]

    results = []
    for right_true, right_false, threshold, in rights:
        acc_true = right_true / len(dists['true'])
        acc_false = right_false / len(dists['false'])
        acc = (right_true + right_false) / (len(dists['true']) + len(dists['false']))

        tp = right_true
        fn = len(dists['true']) - right_true
        tn = right_false
        fp = len(dists['false']) - right_false
        f1 = f1_score(tp, fp, tn, fn)
        results.append([acc_true, tp, fp, acc_false, tn, fn, acc, f1, threshold])

    results_sorted = sorted(results, key=lambda x: x[7], reverse=True)
    print(f'##### Results - {layer} #####')
    print(f'Total Accuracy: {results_sorted[0][6]}')
    print(f'F1 Score: {results_sorted[0][7]}')
    print(f'Best Threshold: {results_sorted[0][8]}')
    df = pd.DataFrame(
        results_sorted,
        columns=['Accuracy_true', 'TP', 'FP', 'Accuracy_false', 'TN', 'FN', 'Total_accuracy', 'F1', 'Threshold']
    )
    df.to_csv(f'results/best_thresholds_{layer}.csv', index=False)

    if results_sorted[0][7] > best_f1:
        best_f1 = results_sorted[0][7]
        best_threshold = results_sorted[0][8]

# ### Save best threshold inside checkpoint
if is_save_best_threshold:
    checkpoint['threshold'] = best_threshold
    torch.save(checkpoint, PATH_MODEL)

    with open(PATH_MODEL.replace('.pth', '_results.json'), 'r+') as file:
        file_json = json.load(file)
        file_json['threshold'] = best_threshold
        file.seek(0)
        json.dump(file_json, file, indent=4)

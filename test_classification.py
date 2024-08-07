"""
Script used to test trained classification model
"""
import os
import json
import pathlib
from tqdm import tqdm

import cv2
import torch
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils.utils import default_transforms


PATH_MODEL = 'checkpoints/20240806-191524/best_97.pth'
TEST_FOLDER = 'dataset/100_sports_image_classification/test'

IS_FLOAT16 = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(PATH_MODEL)
model = checkpoint['model']
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

preds = []
labels = []
folders = os.listdir(TEST_FOLDER)
for idx_folder, folder in enumerate(folders):
    if folder not in checkpoint['id_to_label'].values():
        print(f'Folder {folder} not found in labels of the model... skipping')
        continue
    files = os.listdir(f'{TEST_FOLDER}/{folder}')
    test_tqdm = tqdm(files, desc=f'Testing "{folder}" folder [{idx_folder+1}/{len(folders)}]', leave=False)
    total_files = len(files)
    for idx_filename, filename in enumerate(test_tqdm):
        test_tqdm.set_postfix({f'Files tested': f'{idx_filename+1}/{total_files}'})
        img = transforms(Image.fromarray(cv2.imread(f'{TEST_FOLDER}/{folder}/{filename}')))
        with torch.inference_mode():
            if IS_FLOAT16:
                outputs = model(img.unsqueeze(0).to(device).half())
            else:
                outputs = model(img.unsqueeze(0).to(device))

        result = list(torch.nn.functional.softmax(outputs[0], dim=0).cpu().detach().numpy())
        cls = checkpoint['id_to_label'][result.index(max(result))]
        preds.append(cls)
        labels.append(folder)

# ### Confusion Matrix ###
pathlib.Path('results').mkdir(parents=True, exist_ok=True)
conf_matrix = confusion_matrix(labels, preds)

ticklabels = []
for cls in checkpoint['id_to_label'].values():
    if cls in labels:
        ticklabels.append(cls)

plt.figure(figsize=(max(len(ticklabels), 10), max(len(ticklabels)*0.8, 8)))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=ticklabels,
            yticklabels=ticklabels)
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png')
plt.close()

# ### Accuracy ###
results = {cls: {'correct': 0, 'incorrect': 0, 'accuracy': 0.0} for cls in checkpoint['id_to_label'].values()}
results['total'] = {'correct': 0, 'incorrect': 0, 'accuracy': 0.0}
for idx, cls in enumerate(ticklabels):
    correct = int(conf_matrix[idx, idx])
    incorrect = int(conf_matrix[idx].sum() - conf_matrix[idx, idx])

    results[cls]['correct'] = correct
    results[cls]['incorrect'] = incorrect
    results[cls]['accuracy'] = correct / max(correct + incorrect, 1)

    results['total']['correct'] += correct
    results['total']['incorrect'] += incorrect

results['total']['accuracy'] = results['total']['correct'] / (results['total']['correct'] + results['total']['incorrect'])
with open('results/results_test.json', 'w') as fp:
    json.dump(results, fp)

print(f'Total correct: {results['total']['correct']}')
print(f'Total incorrect: {results['total']['incorrect']}')
print(f'Total accuracy: {results['total']['accuracy']}')
print('Results saved in the "results" folder!')

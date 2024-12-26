import warnings
warnings.filterwarnings("ignore")

# SEEDING
import random, os
import numpy as np
import torch

SEED = 3407
# SEED = 12345
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED) # single GPU, use manual_seed_all for multi GPUs
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(SEED)

from time import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score, roc_auc_score

from lib_data import MyData
from lib_model import SingleTaskNet, SkipResNet

from torchinfo import summary

from os.path import join, isdir
from os import makedirs, listdir

import pandas as pd
import json

import torchio as tio
val_transform = tio.ZNormalization()

img_dir = "../data/lung-window-roi-npy"
task_name = "binary"

# model_name = 'densenet121.ra_in1k'
# model_name = 'resnet10t.c3_in1k'
# model_name = 'tf_efficientnet_b0.ns_jft_in1k'
# model_name = 'tf_efficientnet_b1.ns_jft_in1k'
# model_name = 'tf_efficientnet_b2.ns_jft_in1k'
# model_name = 'tf_efficientnet_b3.ns_jft_in1k'
model_name = 'skipresnet'

model_dir = join("model", task_name, model_name)
n_class = 2

with open("nested_cv_without_indeterminate.json", "r") as f:
    cross_val_split = json.load(f)
annotation = pd.read_csv("dataset_without_indeterminate.csv")

device = "cuda"

val_preds_main = []
val_target_main = []
val_preds_secondary = []
val_target_secondary = []
val_proba_main = []
print(model_dir)
for fold in sorted(listdir(model_dir)):
    if "json" in fold or "." in fold or "csv" in fold:
        continue
    else:
        fold_dir = join(model_dir, fold)
        if model_name == "skipresnet":
            model = SkipResNet(
                n_class=n_class-1
            )
        else:
            model = SingleTaskNet(
                model_name=model_name,
                pretrained=True,
                n_class=n_class-1
            )
        checkpoint = torch.load(join(fold_dir, "checkpoint_best.pth"), map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device)
        for p in model.parameters():
            p.requires_grad = False

        model.eval()
        with torch.no_grad():
            # summary(model, col_names=["num_params", "trainable"])

            case_ids = cross_val_split[str(fold[-1])]["outer_val"]
            val_set = MyData(
                img_dir=img_dir,
                annotation=annotation,
                case_ids=case_ids,
                transform=val_transform
            )
            val_loader = DataLoader(
                dataset=val_set,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                pin_memory=True
            )

            pbar = tqdm(val_loader)
            for step, batch in enumerate(pbar):
                image = batch["data"]
                image = image.to(device)
                                                    
                target_main = batch["target_main"]
                target_main = target_main.to(device)

                preds_main = model(image)

                preds_proba = torch.sigmoid(preds_main)
                val_proba_main.extend(preds_proba.view(-1).cpu().numpy())
                preds_main = preds_proba > 0.5

                preds_main = preds_main.view(-1).cpu().numpy()
                target_main = target_main.view(-1).cpu().numpy()

                val_preds_main.extend(preds_main)
                val_target_main.extend(target_main)
                    
                pbar.set_description(f"evaluating fold {fold[-1]} validation")

labels = list(range(len(np.unique(val_target_main))))
acc = accuracy_score(val_target_main, val_preds_main)
prec = precision_score(val_target_main, val_preds_main, average=None, labels=labels)
rec = recall_score(val_target_main, val_preds_main, average=None, labels=labels)
f1 = f1_score(val_target_main, val_preds_main, average=None, labels=labels)
auc = roc_auc_score(val_target_main, val_proba_main, average=None, labels=labels)
tn, fp, fn, tp = confusion_matrix(val_target_main, val_preds_main).ravel()

print(f"accuracy: {acc:.4f}")
print(f"precision:", ' '.join(f"class {i}: {p:.4f}" for i, p in enumerate(prec)))
print(f"recall:", ' '.join(f"class {i}: {r:.4f}" for i, r in enumerate(rec)))
print(f"f1-score:", ' '.join(f"class {i}: {f:.4f}" for i, f in enumerate(f1)))
print(f"AUC: {auc:.4f}")
print(f"specificity: {tn/(tn+fp):.4f}")

df_metrics = pd.DataFrame(columns=['target', 'prediction', 'probability'])
df_metrics['target'] = val_target_main
df_metrics['prediction'] = val_preds_main
df_metrics['probability'] = val_proba_main
df_metrics.to_csv(join(model_dir, 'metrics.csv'), index=False)
# TO DO: FIX SAVING FOLDER

import warnings
warnings.filterwarnings("ignore")

# SEEDING
import random, os
import numpy as np
import torch

SEED = 3407
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED) # single GPU, use manual_seed_all for multi GPUs
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def SEED_WORKER(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
G = torch.Generator()
G.manual_seed(SEED)

import timm_3d
import torch.optim as optim
from torch.utils.data import DataLoader
from lib_data import MyData, OverSampler
from lib_model import SingleTaskNet, SkipResNet, initialize_weights
from lib_utils import print_to_log_file

### AUGMENTATION ###
import torchio as tio
TRAIN_TRANSFORM = tio.Compose([
    tio.RandomAffine(
        scales=(0.9, 1.1),
        degrees=(-10, 10),
        translation=(-10, 10)
    ),
    tio.ZNormalization()
])
VAL_TRANSFORM = tio.ZNormalization()

from time import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import pandas as pd
from os import makedirs
from os.path import join
import json
import argparse

INDETERMINATE = False
ANNOTATION_FILE = f"dataset_without_indeterminate.csv"
CV_SPLIT_FILE = f"nested_cv_without_indeterminate.json"

ANNOTATION = pd.read_csv(ANNOTATION_FILE)[["Filename", "Malignancy_2", "Malignancy_4"]]
N_CLASS = 2

IMG_DIR = "../data/lung-window-roi-npy"

KFOLD = [0, 1, 2, 3, 4]
NUM_EPOCHS = 50

DEVICE = "cuda"
PRETRAINED = True

BATCH_SIZE = 20
NUM_WORKERS = 0

DROP_RATE = 0.2
INITIAL_LR = 1e-3
WEIGHT_DECAY = 1e-2
USE_LR_SCHEDULER = True
WEIGHTED_LOSS = True

# MODEL_NAME = 'densenet121.ra_in1k'
# MODEL_NAME = 'resnet10t.c3_in1k'
# MODEL_NAME = 'tf_efficientnet_b0.ns_jft_in1k'
# MODEL_NAME = 'tf_efficientnet_b1.ns_jft_in1k'
# MODEL_NAME = 'tf_efficientnet_b2.ns_jft_in1k'
# MODEL_NAME = 'tf_efficientnet_b3.ns_jft_in1k'
MODEL_NAME = 'skipresnet'

BCE_WEIGHT = torch.tensor(178/217).to(DEVICE)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=BCE_WEIGHT)

timestamp = datetime.now()
timestamp = "%d_%d_%d_%02.0d_%02.0d_%02.0d" % (timestamp.year, timestamp.month, timestamp.day,
                                               timestamp.hour, timestamp.minute, timestamp.second)
model_dir = join("model", "binary", MODEL_NAME)
makedirs(model_dir, exist_ok=True)

config = {}
config["n_class"] = N_CLASS
config["indeterminate"] = INDETERMINATE
config["annotation_file"] = ANNOTATION_FILE
config["cv_split_file"] = CV_SPLIT_FILE
config["img_dir"] = IMG_DIR

config["kfold"] = KFOLD
config["num_epochs"] = NUM_EPOCHS
config["device"] = DEVICE
config["pretrained"] = PRETRAINED
config["batch_size"] = BATCH_SIZE
config["num_workers"] = NUM_WORKERS

config["drop_rate"] = DROP_RATE
config["initial_lr"] = INITIAL_LR
config["weight_decay"] = WEIGHT_DECAY
config["use_lr_scheduler"] = USE_LR_SCHEDULER
config["weighted_loss"] = WEIGHTED_LOSS
config["model_name"] = MODEL_NAME

if BCE_WEIGHT is not None: config["bce_weight"] = BCE_WEIGHT.cpu().numpy().tolist()

with open(join(model_dir, "config.json"), "w") as f:
    json.dump(config, f)

with open(CV_SPLIT_FILE, "r") as f:
    cross_val_split = json.load(f)
data_set = {}
data_loader = {}
for outer_fold in KFOLD:
    data_set[outer_fold] = {}
    data_loader[outer_fold] = {}    
    for inner_fold in [0]:
        data_set[outer_fold][inner_fold] = {}
        data_loader[outer_fold][inner_fold] = {}
        for phase in ["inner_train", "inner_val"]:
            CASE_IDS = cross_val_split[str(outer_fold)][str(inner_fold)][phase]
            if "train" in phase:
                data_set[outer_fold][inner_fold][phase] = MyData(
                    img_dir=IMG_DIR,
                    annotation=ANNOTATION,
                    case_ids=CASE_IDS,
                    transform=TRAIN_TRANSFORM
                )
                data_loader[outer_fold][inner_fold][phase] = DataLoader(
                    dataset=data_set[outer_fold][inner_fold][phase],
                    batch_sampler=OverSampler(
                        annotation=ANNOTATION,
                        case_ids=CASE_IDS,
                        batch_size=BATCH_SIZE,
                        n_class=4
                    ),
                    num_workers=NUM_WORKERS,
                    worker_init_fn=SEED_WORKER,
                    generator=G,
                    pin_memory=True
                )
            else:
                data_set[outer_fold][inner_fold][phase] = MyData(
                    img_dir=IMG_DIR,
                    annotation=ANNOTATION,
                    case_ids=CASE_IDS,
                    transform=VAL_TRANSFORM
                )
                data_loader[outer_fold][inner_fold][phase] = DataLoader(
                    dataset=data_set[outer_fold][inner_fold][phase],
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    worker_init_fn=SEED_WORKER,
                    generator=G,
                    shuffle=False,
                    pin_memory=True
                )

def run_outer_val(model):
    model.eval()        
    with torch.no_grad():
        CASE_IDS = cross_val_split[str(outer_fold)]["outer_val"]
        val_set = MyData(
            img_dir=IMG_DIR,
            annotation=ANNOTATION,
            case_ids=CASE_IDS,
            transform=VAL_TRANSFORM
        )
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=1,
            num_workers=NUM_WORKERS,
            shuffle=False,
            pin_memory=True
        )
        
        val_f1_main = []
        val_preds_main = []
        val_target_main = []
        pbar = tqdm(val_loader)
        for step, batch in enumerate(pbar):            
            image = batch["data"]
            image = image.to(DEVICE)

            target_main = batch["target_main"]
            target_main = target_main.to(DEVICE)

            preds_main = model(image)
            preds_main = torch.sigmoid(preds_main) > 0.5
                
            preds_main = preds_main.view(-1).cpu().numpy()
            target_main = target_main.view(-1).cpu().numpy()       
            
            val_preds_main.extend(preds_main)
            val_target_main.extend(target_main)
            
            pbar.set_description(f"running outer fold {outer_fold} validation")
            
        val_f1_main = f1_score(val_preds_main, val_target_main, average=None, labels=list(range(len(best_val_f1))))
        val_f1_main_weighted = f1_score(val_preds_main, val_target_main, average="weighted", labels=list(range(len(best_val_f1))))
        print_to_log_file(log_file, f"outer fold {outer_fold} validation mean f1 main:", ' '.join(f"class {i}: {f1:.4f}" for i, f1 in enumerate(val_f1_main)))
        print_to_log_file(log_file, f"outer fold {outer_fold} validation mean f1 main weighted: {val_f1_main_weighted}")

for outer_fold in KFOLD:
    fold_dir = join(model_dir, f"fold{outer_fold}")
    makedirs(fold_dir, exist_ok=True)
    
    timestamp = datetime.now()
    log_file = join(fold_dir, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second))

    for inner_fold in [0]:
        if MODEL_NAME == "skipresnet":
            model = SkipResNet(
                in_channels=3,
                n_class=N_CLASS-1
            ).to(DEVICE)
            initialize_weights(model)
        else:
            model = SingleTaskNet(
                model_name=MODEL_NAME,
                pretrained=PRETRAINED,
                n_class=N_CLASS-1,
                drop_rate=DROP_RATE
            ).to(DEVICE)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=INITIAL_LR,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=WEIGHT_DECAY,
        )

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
        
        best_val_loss = 1e10
        best_val_f1 = [0. for _ in range(N_CLASS)]
        epochs = range(NUM_EPOCHS)

        epoch_time = [0. for _ in epochs]
        mean_train_loss = [1e10 for _ in epochs]
        mean_val_loss = [1e10 for _ in epochs]
        train_f1_main = [0. for _ in epochs]
        val_f1_main = [0. for _ in epochs]

        start = time()
        print_to_log_file(log_file, MODEL_NAME)
        for epoch in epochs:
            print_to_log_file(log_file, "")
            print_to_log_file(log_file, f"outer fold {outer_fold}, inner fold {inner_fold}, epoch {epoch+1}/{NUM_EPOCHS}")
            
            current_lr = optimizer.param_groups[0]["lr"]
            print_to_log_file(log_file, f"learning rate: {current_lr:.8f}")

            # TRAIN
            model.train()
            pbar = tqdm(data_loader[outer_fold][inner_fold]["inner_train"])
            train_loss = [1e10 for _ in range(len(pbar))]
            epoch_preds_main = []
            epoch_target_main = []
        
            for step, batch in enumerate(pbar):
                image = batch["data"]
                image = image.to(DEVICE)
                
                target_main = batch["target_main"]
                target_main = target_main.to(DEVICE)

                optimizer.zero_grad()
                preds_main = model(image)
                
                loss = criterion(preds_main, target_main.float())
                preds_main = torch.sigmoid(preds_main) > 0.5

                train_loss[step] = loss.item()
                
                loss.backward()
                optimizer.step()
                
                preds_main = preds_main.view(-1).cpu().numpy()
                target_main = target_main.view(-1).cpu().numpy()
                
                epoch_preds_main.extend(preds_main)
                epoch_target_main.extend(target_main)
                    
                pbar.set_description(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, train loss: {loss.item():.4f}")
            
            mean_train_loss[epoch] = sum(train_loss) / len(train_loss)
            train_f1_main[epoch] = f1_score(epoch_preds_main, epoch_target_main, average=None, labels=list(range(len(best_val_f1))))
            
            # VALIDATION    
            model.eval()
            pbar = tqdm(data_loader[outer_fold][inner_fold]["inner_val"])
            val_loss = [1e10 for _ in range(len(pbar))]
            
            with torch.no_grad():
                epoch_preds_main = []
                epoch_target_main = []
                for step, batch in enumerate(pbar):                    
                    image = batch["data"]
                    image = image.to(DEVICE)

                    target_main = batch["target_main"]
                    target_main = target_main.to(DEVICE)

                    preds_main = model(image)
        
                    loss = criterion(preds_main, target_main.float())
                    preds_main = torch.sigmoid(preds_main) > 0.5

                    val_loss[step] = loss.item()
                    
                    preds_main = preds_main.view(-1).cpu().numpy()
                    target_main = target_main.view(-1).cpu().numpy()         
                    
                    epoch_preds_main.extend(preds_main)
                    epoch_target_main.extend(target_main)
                    
                    pbar.set_description(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, val loss: {loss.item():.4f}")
        
                mean_val_loss[epoch] = sum(val_loss) / len(val_loss)
                val_f1_main[epoch] = f1_score(epoch_preds_main, epoch_target_main, average=None, labels=list(range(len(best_val_f1))))
                val_f1_main_weighted = f1_score(epoch_preds_main, epoch_target_main, average='macro', labels=list(range(len(best_val_f1))))
            
            if USE_LR_SCHEDULER:
                lr_scheduler.step()
            
            print_to_log_file(log_file, f"train mean loss: {mean_train_loss[epoch]:.4f}")
            print_to_log_file(log_file, f"val mean loss: {mean_val_loss[epoch]:.4f}")
            print_to_log_file(log_file, f"train mean f1 main:", ' '.join(f"class {i}: {f1:.4f}" for i, f1 in enumerate(train_f1_main[epoch])))
            print_to_log_file(log_file, f"val mean f1 main:", ' '.join(f"class {i}: {f1:.4f}" for i, f1 in enumerate(val_f1_main[epoch])))
            
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_f1": best_val_f1
            }
            torch.save(checkpoint, join(fold_dir, "checkpoint_latest.pth"))
            
            if epoch > 0 and ((abs(mean_val_loss[epoch] - best_val_loss) <= 0.1) or (mean_val_loss[epoch] <= best_val_loss)) and val_f1_main_weighted >= best_val_f1[0]:
                print_to_log_file(log_file, f"new best val weighted f1 main: {best_val_f1[0]:.4f} -> {val_f1_main_weighted:.4f}")
                print_to_log_file(log_file, f"new best val loss: {best_val_loss:.4f} -> {mean_val_loss[epoch]:.4f}")

                best_epoch = epoch
                best_val_loss = mean_val_loss[epoch]
                best_val_f1[0] = val_f1_main_weighted
                
                checkpoint["best_epoch"] = best_epoch
                checkpoint["best_val_loss"] = best_val_loss
                checkpoint["best_val_f1"] = best_val_f1
                torch.save(checkpoint, join(fold_dir, "checkpoint_best.pth"))
    
        total_time = time() - start
        print_to_log_file(log_file, "")
        print_to_log_file(log_file, f"running {epoch + 1} epochs took a total of {total_time:.2f} seconds")
                
    # ACTUAL VALIDATION
    checkpoint = torch.load(join(fold_dir, "checkpoint_best.pth"), map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    
    run_outer_val(model)
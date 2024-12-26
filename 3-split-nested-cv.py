import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import glob
from os.path import join, basename
import json

SEED = 12345
def generate_crossval_split(data, target, seed=SEED, n_splits=5):
    splits = {}
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_indices, test_idx) in enumerate(kfold.split(data, target)):
        outer_train_keys = data[train_indices]
        outer_train_target = target[train_indices]
        outer_val_keys = data[test_idx]
        
        splits[i] = {}
        splits[i]["outer_train"] = list(outer_train_keys)
        splits[i]["outer_val"] = list(outer_val_keys)
        
        for j, (train_idx, val_idx) in enumerate(kfold.split(outer_train_keys, outer_train_target)):
            inner_train_keys = outer_train_keys[train_idx]
            inner_val_keys = outer_train_keys[val_idx]
            splits[i][j] = {}
            splits[i][j]["inner_train"] = list(inner_train_keys)
            splits[i][j]["inner_val"] = list(inner_val_keys)
        
    return splits

print("without indeterminate label")
df = pd.read_csv('dataset_without_indeterminate.csv')[['Filename', 'Malignancy_4']]
data = df['Filename'].values
target = df['Malignancy_4'].values
splits = generate_crossval_split(data, target, seed=SEED, n_splits=5)

for outer_fold in [0,1,2,3,4]:
    outer_train_cases = splits[outer_fold]["outer_train"]
    outer_val_cases = splits[outer_fold]["outer_val"]
    
    for inner_fold in [0,1,2,3,4]:
        inner_train_cases = splits[outer_fold][inner_fold]["inner_train"]
        inner_val_cases = splits[outer_fold][inner_fold]["inner_val"]

    print(f"outer fold: {outer_fold}, train: {len(outer_train_cases)}, val: {len(outer_val_cases)}")
    
with open("nested_cv_without_indeterminate.json", "w") as f:
    json.dump(splits, f)
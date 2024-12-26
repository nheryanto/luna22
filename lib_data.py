from os.path import join
from torch.utils.data import Dataset
import numpy as np
import torchio as tio

# SEEDING
import random, os
import torch

SEED = 12345
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED) # single GPU, use manual_seed_all for multi GPUs
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
class MyData (Dataset): 
    def __init__(self, img_dir, case_ids, annotation, transform=None):        
        self.img_dir = img_dir
        self.case_ids = case_ids
        self.annotation = annotation
        self.transform = transform
        
    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]

        # load modalities & ground truth segmentation
        case_id_npy = case_id.replace(".nii.gz", ".npy")
        img = np.load(join(self.img_dir, case_id_npy), 'r', allow_pickle=True)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, repeats=3, axis=0)
        if self.transform:
            img = self.transform(img)
        
        target_main = self.annotation[self.annotation['Filename'] == case_id][f'Malignancy_2'].values
        data = {
            "data": img.astype(np.float32),
            "target_main": target_main,
            "index": index,
            "case_id": case_id
        }
        
        return data
    
from torch.utils.data import BatchSampler, SubsetRandomSampler
from math import ceil

class OverSampler (BatchSampler):
    def __init__(self, annotation, case_ids, batch_size, n_class=4):
        self.labels = annotation[annotation['Filename'].isin(case_ids)][f'Malignancy_{n_class}'].values
        self.batch_size = batch_size

        # get unique classes and their counts
        self.classes, counts = np.unique(self.labels, return_counts=True)
        
        # sort classes by frequency (descending order)
        sorted_indices = np.argsort(counts)[::-1]
        self.classes = self.classes[sorted_indices]
        self.counts = counts[sorted_indices]
        
        # calculate the number of batches needed to balance classes
        self.num_classes = len(self.classes)
        self.batch_size_per_class = batch_size // self.num_classes
        self.length = ceil(max(self.counts) / self.batch_size_per_class)
        
        # create a dictionary with indices for each class
        self.class_indices_dict = {cls: np.flatnonzero(self.labels == cls) for cls in self.classes}

    def __iter__(self):
        while True:
            batch = []
            for cls in self.classes:
                # sample indices for current class
                indices = SubsetRandomSampler(range(len(self.class_indices_dict[cls])))
                indices = list(indices)
                sampled_indices = np.random.choice(indices, min(len(indices), self.batch_size_per_class), replace=False)
                batch.extend(self.class_indices_dict[cls][sampled_indices])
            
            # shuffle and yield the batch
            random.shuffle(batch)
            yield batch
            
            # stop if the number of batches has been reached
            self.length -= 1
            if self.length <= 0:
                self.length = ceil(max(self.counts) / self.batch_size_per_class)
                break

    def __len__(self):
        return self.length
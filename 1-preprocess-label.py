import numpy as np
import pandas as pd

def flatten_dict(d, num_elements=4):
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, list):
            value = value + [0] * (num_elements - len(value))
            for i, item in enumerate(value):
                flat_dict[f'{key}_{i+1}'] = item
        else:
            flat_dict[key] = value
    return flat_dict

# get consensus of malignancy, texture, and calcification based on median as suggested in https://zenodo.org/records/6559584
def ceil_median_consensus(row, columns):
    ratings = row[columns]
    return np.ceil(np.median(ratings.dropna()))

# get diameter averaged like in 10.1109/ITME56794.2022.00059
def average(row, columns):
    ratings = row[columns]
    return np.mean(ratings.dropna())

if __name__ == "__main__":
    data = np.load('LIDC-IDRI_1176.npy', allow_pickle=True)
    flattened_data = [flatten_dict(d) for d in data]
    df = pd.DataFrame(flattened_data)
    
    # TEXTURE
    columns = [f'Texture_{i}' for i in range(1,9)]
    df['Texture'] = df.apply(lambda row: ceil_median_consensus(row, columns), axis=1)
    
    # CALCIFICATION
    columns = [f'Calcification_{i}' for i in range(1,9)]
    df['Calcification'] = df.apply(lambda row: ceil_median_consensus(row, columns), axis=1)
    
    # DIAMETER
    columns = [f'Diameter_{i}' for i in range(1,9)]
    df['Diameter'] = df.apply(lambda row: average(row, columns), axis=1)
    
    # MALIGNANCY
    columns = [f'Malignancy_{i}' for i in range(1,9)]
    df['Malignancy'] = df.apply(lambda row: ceil_median_consensus(row, columns), axis=1)
    
    dataset = df[[
        'Filename', 'Malignancy', 'Texture', 'Calcification',
        'Diameter', 'VoxelCoordX', 'VoxelCoordY', 'VoxelCoordZ',
        'SeriesInstanceUID'
    ]].reset_index(drop=True)
    dataset['Malignancy_5'] = dataset['Malignancy'] - 1
    malignancy_2 = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
    
    ### WITHOUT INDETERMINATE ###
    dataset_without_indeterminate = dataset[dataset['Malignancy_5'] != 2].copy()
    dataset_without_indeterminate['Malignancy_2'] = dataset_without_indeterminate['Malignancy_5'].map(malignancy_2)
    
    malignancy_4 = {0: 0, 1: 1, 3: 2, 4: 3}
    dataset_without_indeterminate['Malignancy_4'] = dataset_without_indeterminate['Malignancy_5'].map(malignancy_4)
    dataset_without_indeterminate.drop(columns=['Malignancy_5'], inplace=True)
    dataset_without_indeterminate.to_csv("dataset_without_indeterminate.csv", index=False)
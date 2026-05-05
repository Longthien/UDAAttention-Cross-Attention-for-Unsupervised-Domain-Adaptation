from pathlib import Path
import numpy as np
import os

OEM_DATA_DIR = '/home/ubuntu/thien/dataset/OpenEarthMap/OpenEarthMap_wo_xBD/'
TRAIN_LIST = '/home/ubuntu/thien/dataset/OpenEarthMap/OpenEarthMap_wo_xBD/train.txt'
VAL_LIST = '/home/ubuntu/thien/dataset/OpenEarthMap/OpenEarthMap_wo_xBD/val.txt'
TEST_LIST = '/home/ubuntu/thien/dataset/OpenEarthMap/OpenEarthMap_wo_xBD/test.txt'

fns = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/labels/" in str(f)]
print("total labels: ", len(fns))

# Get all .tif files under images folders
fns = [f for f in Path(OEM_DATA_DIR).rglob("*.tif") if "/images/" in str(f)]
print("total images: ", len(fns))
# Load filenames from train/val lists
train_names = np.loadtxt(TRAIN_LIST, dtype=str)
val_names = np.loadtxt(VAL_LIST, dtype=str)
test_names = np.loadtxt(TEST_LIST, dtype=str)
print(len(train_names))

# Match files by filename only
train_fns = [str(f) for f in fns if f.name in train_names]
val_fns   = [str(f) for f in fns if f.name in val_names]
test_fns = [str(f) for f in fns if f.name in test_names]
# print("Total samples      :", len(fns))
# print("Training samples   :", len(train_fns))
# print("Validation samples :", len(val_fns))
# train_fns = [str(f.name) for f in fns if f.name in train_names]
# val_fns   = [str(f.name) for f in fns if f.name in val_names]
# test_fns = [str(f.name) for f in fns if f.name in test_names]

# available_fs=[]

# available_fs = train_fns + val_fns + test_fns
# print('Available samples:',len(available_fs))
# # print(available_fs)
# missing = []
# for f in test_fns:
#     if f in available_fs:
#         missing.append(f)
# print(len(missing))
# train_missings = [str(f) for f in fns if ((f.name not in train_names) and (f.name not in val_names) and (f.name not in test_names))]
# print(train_missings)
print("Total samples      :", len(fns))
print("Training samples   :", len(train_fns))
print("Validation samples :", len(val_fns))

# Convert to relative paths (remove OEM_DATA_DIR)
train_rel = [os.path.relpath(f, OEM_DATA_DIR) for f in train_fns]
val_rel   = [os.path.relpath(f, OEM_DATA_DIR) for f in val_fns]
train_rel = [f.replace('.tif','') for f in train_rel]
val_rel   = [f.replace('.tif','') for f in val_rel]
# Write out lists
split_dict = {
    'train': train_rel,
    'val': val_rel,
}

for name, split in split_dict.items():
    out_path = f'configs/split/{name}.txt'
    with open(out_path, 'w') as file:
        file.write('\n'.join(split))
import random
sample_ratio = 0.05
train_sample = random.sample(train_rel, int(len(train_rel) * sample_ratio))

print("Generating train target split...")
out_path = 'configs/split/train_5percent.txt'
with open(out_path, 'w') as file:
    file.write('\n'.join(train_sample))
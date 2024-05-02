import os
import random
import glob
import numpy as np
import torch
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils import *

# from utils import crop_sample, pad_sample, resize_sample, normalize_volume

input_dir= "/Users/vynguyen/Documents/VSCode/Abdominal_segmentation/CT_FAT"
output_dir_images = "/Users/vynguyen/Documents/VSCode/Abdominal_segmentation/preprocess/images"
output_dir_masks = "/Users/vynguyen/Documents/VSCode/Abdominal_segmentation/preprocess/masks"

class CTFatDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 1
    out_channels = 1
    
    def __init__(
        self,
        images_dir,
        transform=None,
        image_size= 256,
        subset="train",
        random_sampling=True,
        validation_cases=4,
        seed= 42,
    ):
        assert subset in ["all", "train", "validation"]

        # get list of image and mask filenames and sort
        # image_filenames = sorted([filename for filename in os.listdir(input_dir) if "_Img.tif" in filename])
        # mask_filenames = sorted([filename for filename in os.listdir(input_dir) if "_Msk.tif" in filename])
        image_filenames = sorted(glob.glob(os.path.join(images_dir, '*_Img.tif')))
        mask_filenames = sorted(glob.glob(os.path.join(images_dir, '*_Msk.tif')))
        n_img = len(image_filenames)

        os.makedirs(output_dir_images, exist_ok = True)
        os.makedirs(output_dir_masks, exist_ok = True)

        images_array = []
        masks_array = []
        
        # select cases to subset 
        if not subset == "all":
            random.seed(seed)
            val_imgfilenames = random.sample(image_filenames, k= validation_cases)
            if subset == "validation":
                input_images = val_imgfilenames
            else:
                input_images = sorted(list(set(image_filenames).difference(val_imgfilenames)))
        else:
            input_images = image_filenames
        
        # read images, masks
        for i in range(len(input_images)):
            img = imread(input_images[i])
            msk = imread(input_images[i].split("_Img.tif")[0] + "_Msk.tif")
            
            # # label mapping: merge labels 1 into label 3 and label 5 into label 10
            # msk[(msk < 5) & (msk > 0)] = 1     # relabel for labels 1 and 3 - outer region
            # msk[(msk >= 5)] = 2    # relabel for labels 5 and 10 - inner region
            msk[(msk >6 )] = 0
            
            images_array.append(img)
            masks_array.append(msk)
        
        self.images_array = images_array
        self.masks_array = masks_array
        self.transform = transform

# def data_loader(args):
#     dataset = Dataset(
#         transform=None,
#         images_dir= input_dir,
#         subset="train",
#         image_size=args.image_size,
#         random_sampling=False,
#     )
#     loader = DataLoader(
#         dataset, batch_size=args.batch_size, drop_last=False, num_workers=1
#     )
#     return loader

    def __len__(self):
        return len(self.images_array)

    def __getitem__(self, idx):
        
        img = MinMaxScaler(self.images_array[idx])
        msk = self.masks_array[idx]
        
        if self.transform:
            img = self.transform(img)
            msk = self.transform(msk)
        
        return img.float(), msk.float()

    
    
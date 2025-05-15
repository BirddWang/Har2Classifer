from torch.utils.data import Dataset
from torchvision.transforms import Compose, Pad, CenterCrop, ToTensor, ToPILImage
import os
import nibabel as nib
import numpy as np
import torch
import random

default_transform = Compose([ToPILImage(), Pad(40), CenterCrop([224, 224])])

def slice_to_tensor(slice):
    # shape: (192, 224)
    p99 = np.percentile(slice, 95)
    slice = slice / (p99 + 1e-5)
    slice = np.clip(slice, a_min=0.0, a_max=1.0)
    slice = np.array(default_transform(slice))
    slice = ToTensor()(slice)
    return slice

def get_tensor_from_fpath(fpath):
    if os.path.exists(fpath):
        img = nib.load(fpath).get_fdata().astype(np.float32).transpose(0, 2, 1)
        # shape: (192, 224, 192)
        tensor_img = torch.empty(0)
        for slice in img:
            slice = slice_to_tensor(slice)
            tensor_img = torch.cat((tensor_img, slice), dim=0)
    else:
        tensor_img = torch.ones([192, 224, 224])
    return tensor_img

def background_mask(img):
    # shape: (192, 224, 224)
    masks = torch.empty(0)
    for slice in img:
        # shape: (224, 224)
        mask = (torch.ones([224, 224]) * slice.ge(0.15)).bool()
        masks = torch.cat((masks, mask.unsqueeze(0)), dim=0)
    return masks

def ADNI_BASE_AD():
    dir = 'data/ADNI/AD-BASE-prep/'
    return [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith('.nii.gz')]
def ADNI_BASE_CN():
    dir = 'data/ADNI/CN-BASE-prep/'
    return [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith('.nii.gz')]


class ADNI(Dataset):
    def __init__(self):
        super(ADNI, self).__init__()
        self.img_paths = ADNI_BASE_AD() + ADNI_BASE_CN()
        random.shuffle(self.img_paths)

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = 1 if 'AD-' in img_path else 0
        img = get_tensor_from_fpath(img_path)
        mask = background_mask(img)
        masked_img = img * mask
        return {
            "masked_image": masked_img,
            "label": label,
            "mask": mask,
        }
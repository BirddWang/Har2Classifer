from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import nibabel as nib
import numpy as np
import torch
import random
import torchio as tio
import hashlib
import pickle as pkl

transform = tio.Compose([
    tio.RandomFlip(axes=('LR',), p=0.5),
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
    tio.RandomElasticDeformation(p=0.2),
    tio.RandomNoise(p=0.3),
    tio.CropOrPad((192, 224, 224), padding_mode='constant'),
    tio.RescaleIntensity((0, 1))
])

def get_cache_path(fpath):
    # Create a unique cache path based on the file path
    hash_object = hashlib.md5(fpath.encode())
    cache_path = f"tmp/{hash_object.hexdigest()}.pt"
    return cache_path

def get_dir_size_gb(dir_path):
    total_size = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 ** 3)  # Convert bytes to gigabytes


def background_mask(img):
    # shape: (192, 224, 224)
    masks = torch.empty(0)
    for slice in img:
        # shape: (224, 224)
        mask = (torch.ones([224, 224]) * slice.ge(0.15)).bool()
        masks = torch.cat((masks, mask.unsqueeze(0)), dim=0)
    return masks


class ADNI(Dataset):
    def __init__(self, training=True, dir = "./data/ADNI", data_type = "harmonized"):
        super(ADNI, self).__init__()
        if data_type == "harmonized":
            ad_paths = [os.path.join(dir, data_type, "ad", path) for path in os.listdir(os.path.join(dir, data_type, "ad")) if path.endswith("_fusion.nii.gz")]
            cn_paths = [os.path.join(dir, data_type, "cn", path) for path in os.listdir(os.path.join(dir, data_type, "cn")) if path.endswith("_fusion.nii.gz")]
        elif data_type == "preprocessed":
            ad_paths = [os.path.join(dir, data_type, "ad", path) for path in os.listdir(os.path.join(dir, data_type, "ad")) if path.endswith("_prep.nii.gz")]
            cn_paths = [os.path.join(dir, data_type, "cn", path) for path in os.listdir(os.path.join(dir, data_type, "cn")) if path.endswith("_prep.nii.gz")]
        elif data_type == "beta":
            ad_paths = [os.path.join(dir, data_type, "ad", path) for path in os.listdir(os.path.join(dir, data_type, "ad")) if path.endswith("_beta.pkl")]
            cn_paths = [os.path.join(dir, data_type, "cn", path) for path in os.listdir(os.path.join(dir, data_type, "cn")) if path.endswith("_beta.pkl")]

        print(f"Found {len(ad_paths)} AD and {len(cn_paths)} CN images in {data_type} dataset")
        self.img_paths = ad_paths + cn_paths
        self.len_ad = len(ad_paths)
        self.len_cn = len(cn_paths)
        self.labels = {path:1 for path in ad_paths}
        self.labels.update({path:0 for path in cn_paths})
        self.data_type = data_type

        random.shuffle(self.img_paths)

    def slice_to_tensor(self, slice):
        # shape: (192, 224)
        p99 = np.percentile(slice, 95)
        slice = slice / (p99 + 1e-5)
        slice = np.clip(slice, a_min=0.0, a_max=1.0)
        slice = ToTensor()(slice)
        return slice

    def get_tensor_from_fpath(self, fpath):
        if os.path.exists(get_cache_path(fpath)):
            try:
                tensor_img = torch.load(get_cache_path(fpath))
                tensor_img = transform(torch.unsqueeze(tensor_img, dim=0))
                return tensor_img 
                # print(f"Loaded cached {fpath}")
            except Exception as e:
                print(f"Error loading cached {fpath}: {e}")
            # tensor_img.cuda()
            
        if os.path.exists(fpath):
            if self.data_type != "beta":
                try: 
                    img = nib.load(fpath).get_fdata().astype(np.float32).transpose(0, 2, 1)
                    tensor_img = torch.empty(0)
                    for slice in img:
                        slice = self.slice_to_tensor(slice)
                        tensor_img = torch.cat((tensor_img, slice), dim=0)
                except Exception as e:
                    print(f"Error loading {fpath}: {e}")
                    return torch.ones([1, 192, 224, 224])
            # shape: (192, 224, 192)
            # print(f"Loading {fpath} with shape {img.shape}")
            else:
                with open(fpath, 'rb') as f:
                    tensor_img = pkl.load(f)
                    tensor_img = torch.tensor(tensor_img, dtype=torch.float32)
            print(f"Loaded {fpath} with shape {tensor_img.shape}")
        else:
            tensor_img = torch.ones([192, 224, 224])
            print("Error: File not found", fpath)

        
        if get_dir_size_gb("tmp") < 100:
            torch.save(tensor_img, get_cache_path(fpath))
            # print(f"cached {fpath}")
        # tensor_img = tensor_img.cuda() 
        tensor_img = transform(torch.unsqueeze(tensor_img, dim=0))
        return tensor_img

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # print(f"Getting item {idx} from dataset")
        # # flush print buffer
        # import sys
        # sys.stdout.flush()
        img_path = self.img_paths[idx]
        label = self.labels[img_path]
        img = self.get_tensor_from_fpath(img_path)
        return {
            "image": img, 
            "label": label,
        }
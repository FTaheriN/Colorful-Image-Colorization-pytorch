import numpy as np

from PIL import Image
from skimage import color

import sklearn.neighbors as skl_nn

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if(out_np.ndim==2):
        out_np = np.tile(out_np[:,:,None],3)
    return out_np


def resize_img(img, HW=(256,256), resample=3):
	  return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))
 

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    return img_lab_rs


class ImageDataset(Dataset):
    def __init__(self, imgs_list, transforms=None):
        super(ImageDataset, self).__init__()
        self.imgs_list = imgs_list
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.imgs_list[index]   

        img = load_img(image_path)
        img_lab_rs = preprocess_img(img[:,:,:3])

        img_l_rs = img_lab_rs[:,:,0]
        tens_l_rs = torch.Tensor(img_l_rs)[None,:,:]

        img_ab_rs = img_lab_rs[:,:,1:]
        # tens_ab_rs = torch.Tensor(img_ab_rs)[:,:,:]torch.permute(x, (2, 0, 1))
        # y = groundTruth_soft_encodeing(img_ab_rs, self.nn_finder)

        return tens_l_rs, img_ab_rs.transpose(2, 0, 1)


    def __len__(self):
        return len(self.imgs_list)
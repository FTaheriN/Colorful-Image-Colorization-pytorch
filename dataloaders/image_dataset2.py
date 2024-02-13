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

def groundTruth_soft_encodeing(img_ab_rs, n_neighbor):
    h, w = img_ab_rs.shape[:2]
    img_ab = np.vstack((img_ab_rs[:,:,0].reshape(h*w), img_ab_rs[:,:,1].reshape(h*w))).T

    neigh_dist, neigh_idx = n_neighbor.kneighbors(img_ab)

    # weighting 5 nearest neightbors with gaussian kernel
    sigma = 5
    neighbor_weights = np.exp(-np.power(neigh_dist, 2) / 2*np.power(sigma,2))
    neighbor_weights = neighbor_weights / np.sum(neighbor_weights, axis=1).reshape(h*w, -1)

    # build the expectedd output
    y = np.zeros((h*w, 313))
    index = np.arange(h*w).reshape(h*w, -1)

    y[index, neigh_idx] = neighbor_weights

    return torch.permute((torch.tensor(y.reshape(h, w, 313))),(2,0,1))


class ImageDataset2(Dataset):
    def __init__(self, imgs_list, transforms=None):
        super(ImageDataset2, self).__init__()
        self.imgs_list = imgs_list
        self.transforms = transforms

        # Load the array of quantized ab value
        q_ab = np.load("./pts_in_hull.npy")
        self.nb_q = q_ab.shape[0]
        # Fit a NN to q_ab
        self.nn_finder = skl_nn.NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(q_ab)

    def __getitem__(self, index):
        image_path = self.imgs_list[index]   

        img = load_img(image_path)
        img_lab_rs = preprocess_img(img[:,:,:3])

        img_l_rs = img_lab_rs[:,:,0]
        tens_l_rs = torch.Tensor(img_l_rs)[None,:,:]

        img_ab_rs = img_lab_rs[:,:,1:].astype(np.int32)
        # tens_ab_rs = torch.Tensor(img_ab_rs)[:,:,:]torch.permute(x, (2, 0, 1))
        y = groundTruth_soft_encodeing(img_ab_rs, self.nn_finder)

        return tens_l_rs, y


    def __len__(self):
        return len(self.imgs_list)
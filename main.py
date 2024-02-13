import time
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

import torch 

import plots
from model import colorizationCNN
from dataloaders import load_images
from deeplearning.train import train
from deeplearning.test import test
from utils import read_yaml_config

############################## Reading Model Parameters ##############################
config = read_yaml_config()
# data_path = config['dataset1']['path'] # change to dataset2 for the regression task
learning_rate = config['learning_rate']
epochs = config['epochs']
batch_size = config['batch_size']
gamma = config['gamma']
step_size = config['step_size']
ckpt_save_freq = config['ckpt_save_freq']

#################################### Loading Data ####################################


####################################      Main     #################################### 

def main():
    train_loader, valid_loader, test_loader = load_images(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    l2_colorizatio_model = colorizationCNN()

    trainer = train(
        train_loader=train_loader,
        val_loader=test_loader,
        model = l2_colorizatio_model,
        model_name="l2_colorizatio_model",
        epochs=epochs,
        learning_rate=learning_rate,
        gamma = gamma,
        step_size = step_size,
        device=device,
        load_saved_model=False,
        ckpt_save_freq=ckpt_save_freq,
        ckpt_save_path="/content/drive/MyDrive/MSC/DeepLearning/HW3/ckpt/",
        ckpt_path="/content/drive/MyDrive/MSC/DeepLearning/HW3/ckpt/ckpt_l2_colorizatio_model.ckpt",
        report_path="/content/drive/MyDrive/MSC/DeepLearning/HW3/rep/")

main()
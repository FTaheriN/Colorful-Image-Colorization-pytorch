import torch 
import os 
import glob
import random

import dataloaders


def load_images(batch_size):
    DIR = "/content/landscapes/"

    images = []
    for image in os.listdir(DIR):
        images .append(os.path.normpath(DIR + image))

    random.shuffle(images)

    train_data = images[:int(0.8*len(images))]
    valid_data = images[int(0.8*len(images)):int(0.9*len(images))]
    test_data  = images[int(0.9*len(images)):]

    print("train: {}, valid: {}, test: {}".format(len(train_data), len(valid_data), len(test_data)))

    train_dataset = dataloaders.ImageDataset(imgs_list = train_data)
    valid_dataset = dataloaders.ImageDataset(imgs_list = valid_data)
    test_dataset  = dataloaders.ImageDataset(imgs_list = test_data)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)

    return train_loader, valid_loader, test_loader
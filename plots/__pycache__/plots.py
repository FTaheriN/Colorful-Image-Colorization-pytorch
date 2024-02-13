import torch 
import numpy as np
from skimage import color, io

import matplotlib.pyplot as plt
plt.ion()

import torchvision.utils
model.double()
with torch.no_grad():

    # pick a random subset of images from the test set
    image_inds = np.random.choice(len(test_dataset), 9, replace=False)
    print(image_inds)
    lab_batch = torch.stack([torch.cat((test_dataset[i][0],torch.from_numpy(test_dataset[i][1]))) for i in image_inds])
        
    print(lab_batch.shape)
    lab_batch = lab_batch.to(device)

    # predict colors (ab channels)
    predicted_ab_batch = model(lab_batch[:, 0:1, :, :])
    predicted_lab_batch = torch.cat([lab_batch[:, 0:1, :, :], predicted_ab_batch], dim=1)

    lab_batch = lab_batch.cpu()
    predicted_lab_batch = predicted_lab_batch.cpu()

    rgb_batch = []
    predicted_rgb_batch = []

    for i in range(lab_batch.size(0)):
        rgb_img = color.lab2rgb(np.transpose(lab_batch[i, :, :, :].numpy().astype('float64'), (1, 2, 0)))
        rgb_batch.append(torch.FloatTensor(np.transpose(rgb_img, (2, 0, 1))))
        predicted_rgb_img = color.lab2rgb(np.transpose(predicted_lab_batch[i, :, :, :].numpy().astype('float64'), (1, 2, 0)))
        predicted_rgb_batch.append(torch.FloatTensor(np.transpose(predicted_rgb_img, (2, 0, 1))))

    # plot images
    fig, ax = plt.subplots(figsize=(15, 15), nrows=1, ncols=2)
    ax[0].imshow(np.transpose(torchvision.utils.make_grid(torch.stack(predicted_rgb_batch), nrow=3).numpy(), (1, 2, 0)))
    ax[0].title.set_text('re-colored')
    ax[1].imshow(np.transpose(torchvision.utils.make_grid(torch.stack(rgb_batch), nrow=3).numpy(), (1, 2, 0)))
    ax[1].title.set_text('original')
    plt.show()



df = pd.read_csv("/content/drive/MyDrive/MSC/DeepLearning/HW3/rep/l2_colorizatio_model_full2.csv")
train_rep1 = df.loc[df['mode'] == 'train'].reset_index().groupby('epoch').mean(numeric_only=True)
valid_rep1 = df.loc[df['mode'] == 'val'].reset_index().groupby('epoch').mean(numeric_only=True)

plt.plot(train_rep1['loss_batch'], color="green")
plt.legend(['train loss'])
plt.plot(valid_rep1['avg_val_loss_till_current_batch'], color="red")
plt.legend(['train loss', 'testloss'])
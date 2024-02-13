import torch
import torch.nn as nn


def test(model, test_loader, device):
    model.to(device)

    model.eval()
    criterion = nn.MSELoss()

    test_loss_avg, num_batches = 0, 0
    for images, labels in test_loader:

        with torch.no_grad():

            images = images.to(device).float()
            labels = labels.to(device)
            labels_pred = model(images)
            
            loss = criterion(labels_pred, labels)

            test_loss_avg += loss.item()
            num_batches += 1

    test_loss_avg /= num_batches
    print('average loss: %f' % (test_loss_avg))
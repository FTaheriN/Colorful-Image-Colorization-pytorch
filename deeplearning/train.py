import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

from utils import load_model, save_model, AverageMeter



def train(
    train_loader,
    val_loader,
    model,
    model_name,
    epochs,
    learning_rate,
    gamma,
    step_size,
    device,
    load_saved_model,
    ckpt_save_freq,
    ckpt_save_path,
    ckpt_path,
    report_path,
):

    model = model.to(device)

    # loss function
    criterion = nn.MSELoss()

    # optimzier
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_saved_model:
        model, optimizer = load_model(
            ckpt_path=ckpt_path, model=model, optimizer=optimizer
        )

    lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "image_type",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_val_loss_till_current_batch"])

    for epoch in range(1, 10):
        loss_avg_train = AverageMeter()
        loss_avg_val = AverageMeter()

        model.train()
        mode = "train"

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_avg_train.update(loss.item(), images.size(0))

            if (batch_idx+1) % 10 == 0:
              print(f"Epoch: {epoch}, batch_index: {batch_idx}, Average loss {loss_avg_train.avg}")

            new_row = pd.DataFrame(
                {"model_name": model_name,
                 "mode": mode,
                 "image_type":"original",
                 "epoch": epoch,
                 "learning_rate":optimizer.param_groups[0]["lr"],
                 "batch_size": images.size(0),
                 "batch_index": batch_idx,
                 "loss_batch": loss.detach().item(),
                 "avg_train_loss_till_current_batch":loss_avg_train.avg,
                 "avg_val_loss_till_current_batch":None},index=[0])
            

            
            report.loc[len(report)] = new_row.values[0]
            # report.to_csv(f"{report_path}/{model_name}_report3.csv")

        save_model(
            file_path=ckpt_save_path,
            file_name=f"ckpt_{model_name}.ckpt",
            model=model,
            optimizer=optimizer,
        )
        model.eval()
        mode = "val"
        with torch.no_grad():
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            for batch_idx, (images, labels) in loop_val:
                optimizer.zero_grad()
                images = images.to(device).float()
                labels = labels.to(device)
                labels_pred = model(images)
                loss = criterion(labels_pred, labels)
                loss_avg_val.update(loss.item(), images.size(0))
                new_row = pd.DataFrame(
                    {"model_name": model_name,
                     "mode": mode,
                     "image_type":"original",
                     "epoch": epoch,
                     "learning_rate":optimizer.param_groups[0]["lr"],
                     "batch_size": images.size(0),
                     "batch_index": batch_idx,
                     "loss_batch": loss.detach().item(),
                     "avg_train_loss_till_current_batch":None,
                     "avg_val_loss_till_current_batch":loss_avg_val.avg},index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    refresh=True,
                )
        
        lr_scheduler.step()
#         report.to_csv(f"{report_path}/{model_name}_epoch{epoch}.csv")
    report.to_csv(f"{report_path}/{model_name}_full2.csv")
    return model, optimizer, report

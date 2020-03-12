# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 02:09:43 2020

@author: zhaog
"""
import os
import argparse
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import QQPDataset
from utils import train, validate
from model import ESIM

def main(train_q1_file, train_q2_file, train_labels_file,
         dev_q1_file, dev_q2_file, dev_labels_file,
         embeddings_file,
         target_dir,
         hidden_size=128,
         dropout=0.5,
         num_classes=2,
         epochs=15,
         batch_size=64,
         lr=0.001,
         patience=5,
         max_grad_norm=10.0,
         gpu_index=0,
         checkpoint=None):
    
    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
    
    print(20 * "=", " Preparing for training ", 20 * "=")
    
    # 保存模型的路径
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_q1 = np.load(train_q1_file)
    train_q2 = np.load(train_q2_file)
    train_labels = np.load(train_labels_file)
#    train_labels = label_transformer(train_labels)
    
    train_data = {"q1": train_q1,
                  "q2": train_q2,
                  "labels": train_labels}
    
    train_data = QQPDataset(train_data)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    print("\t* Loading validation data...")
    dev_q1 = np.load(dev_q1_file)
    dev_q2 = np.load(dev_q2_file)
    dev_labels = np.load(dev_labels_file)
#    dev_labels = label_transformer(dev_labels)
    
    dev_data = {"q1": dev_q1,
                "q2": dev_q2,
                "labels": dev_labels}
    
    dev_data = QQPDataset(dev_data)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    embeddings = torch.tensor(np.load(embeddings_file), dtype=torch.float).to(device)

    model = ESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device).to(device)
    
    
    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    # 过滤出需要梯度更新的参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    optimizer = torch.optim.Adam(parameters, lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.85,
                                                           patience=0)
    
    best_score = 0.0
    start_epoch = 1
    
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
    '''
     # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy = validate(model,
                                             dev_loader,
                                             criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%"
          .format(valid_loss, (valid_accuracy*100)))
    
    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training ESIM model on device: {}".format(device),
          20 * "=")
    '''
    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model,
                                                       train_loader,
                                                       optimizer,
                                                       criterion,
                                                       epoch,
                                                       max_grad_norm)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate(model,
                                                          dev_loader,
                                                          criterion)

        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            # Save the best model. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best model, use the 'esim_*.pth.tar'
            # checkpoints instead.
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                        os.path.join(target_dir, "best.pth.tar"))

        # Save the model at each epoch.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                    os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break
  
    
    # Plotting of the loss curves for the train and validation sets.
    # plt.figure()
    # plt.plot(epochs_count, train_losses, "-r")
    # plt.plot(epochs_count, valid_losses, "-b")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.legend(["Training loss", "Validation loss"])
    # plt.title("Cross entropy loss")
    # # plt.show()
    # plt.savefig('./loss_curves.png')
    
if __name__ == "__main__":
    default_config = "./config.json"

    parser = argparse.ArgumentParser(description="Train the ESIM model on QQP")
    parser.add_argument("--config",
                        default=default_config,
                        help="Path to a json configuration file")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Path to a checkpoint file to resume training")
    parser.add_argument("--gpu",
                        default=0,
                        help="which cuda device to use")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)

    main(os.path.normpath(os.path.join(script_dir, config["train_q1_data"])),
         os.path.normpath(os.path.join(script_dir, config["train_q2_data"])),
         os.path.normpath(os.path.join(script_dir, config["train_labels_data"])),
         os.path.normpath(os.path.join(script_dir, config["dev_q1_data"])),
         os.path.normpath(os.path.join(script_dir, config["dev_q2_data"])),
         os.path.normpath(os.path.join(script_dir, config["dev_labels_data"])),
         os.path.normpath(os.path.join(script_dir, config["embeddings"])),
         os.path.normpath(os.path.join(script_dir, config["target_dir"])),
         config["hidden_size"],
         config["dropout"],
         config["num_classes"],
         config["epochs"],
         config["batch_size"],
         config["lr"],
         config["patience"],
         config["max_gradient_norm"],
         args.gpu,
         args.checkpoint)
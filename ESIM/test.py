# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:46:32 2020

@author: zhaog
"""
import torch
from sys import platform
from torch.utils.data import DataLoader
from data import LCQMC_Dataset, load_embeddings
from model import ESIM
from utils import test

def main(test_file, vocab_file, embeddings_file, pretrained_file, max_length=50, gpu_index=0, batch_size=64):
    """
    Test the ESIM model with pretrained weights on some dataset.
    Args:
        test_file: The path to a file containing preprocessed NLI data.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        vocab_size: The number of words in the vocabulary of the model
            being tested.
        embedding_dim: The size of the embeddings in the model.
        hidden_size: The size of the hidden layers in the model. Must match
            the size used during training. Defaults to 300.
        num_classes: The number of classes in the output of the model. Must
            match the value used during training. Defaults to 3.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location="cuda:0")
    # Retrieving model parameters from checkpoint.
    hidden_size = checkpoint["model"]["projection.0.weight"].size(0)
    num_classes = checkpoint["model"]["classification.6.weight"].size(0)
    embeddings = load_embeddings(embeddings_file)
    print("\t* Loading test data...")    
    test_data = LCQMC_Dataset(test_file, vocab_file, max_length)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    print("\t* Building model...")
    model = ESIM(hidden_size, embeddings=embeddings, num_classes=num_classes, device=device).to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing ESIM model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy = test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%\n".format(batch_time, total_time, (accuracy*100)))


if __name__ == "__main__":
    main("../data/LCQMC_test.csv", "../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar")
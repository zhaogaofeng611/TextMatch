# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:30:07 2020

@author: zhaog
"""
import torch
import argparse
from sys import platform
from torch.utils.data import DataLoader
from data import LCQMC_Dataset, load_embeddings
from model import RE2
from util import test

def main(args, test_file, vocab_file, embeddings_file, pretrained_file, max_length=50, gpu_index=0, batch_size=128):

    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    embeddings = load_embeddings(embeddings_file)
    print("\t* Loading test data...")    
    test_data = LCQMC_Dataset(test_file, vocab_file, max_length)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    print("\t* Building model...")
    model = RE2(args, embeddings, device=device).to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing RE2 model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy, auc = test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%, auc: {:.4f}\n".format(batch_time, total_time, (accuracy*100), auc))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", default=0.2)
    parser.add_argument("--embedding_dim", default=300)
    parser.add_argument("--hidden_size", default=150)
    parser.add_argument("--kernel_sizes", default=[3])
    parser.add_argument("--blocks", default=2)
    parser.add_argument("--fix_embeddings", default=True)
    parser.add_argument("--encoder", default='cnn')
    parser.add_argument("--enc_layers", default=2)
    parser.add_argument("--alignment", default='linear')
    parser.add_argument("--fusion", default='full')
    parser.add_argument("--connection", default='aug')
    parser.add_argument("--prediction", default='full')
    parser.add_argument("--num_classes", default=2)
    args = parser.parse_args()
    
    main(args, "../data/LCQMC_test.csv", "../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar")
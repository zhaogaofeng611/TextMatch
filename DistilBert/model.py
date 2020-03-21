# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 21:06:25 2020

@author: zhaog
"""
import torch
from torch import nn
from transformers import DistilBertForSequenceClassification, DistilBertConfig

class DistilBertModel(nn.Module):
    def __init__(self):
        super(DistilBertModel, self).__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained("adamlin/bert-distil-chinese", num_labels = 2)
        self.device = torch.device("cuda")
        for param in self.distilbert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.distilbert(input_ids = batch_seqs, attention_mask = batch_seq_masks, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
    

class DistilBertModelTest(nn.Module):
    def __init__(self):
        super(DistilBertModelTest, self).__init__()
        config = DistilBertConfig.from_pretrained('models/config.json')
        self.distilbert = DistilBertForSequenceClassification(config)  # /bert_pretrain/
        self.device = torch.device("cuda")

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.distilbert(input_ids = batch_seqs, attention_mask = batch_seq_masks, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
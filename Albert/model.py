# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 21:06:25 2020

@author: zhaog
"""
import torch
from torch import nn
from transformers import AlbertForSequenceClassification, AlbertConfig

class AlbertModel(nn.Module):
    def __init__(self):
        super(AlbertModel, self).__init__()
        self.albert = AlbertForSequenceClassification.from_pretrained("voidful/albert_chinese_base", num_labels = 2)  # /bert_pretrain/
        self.device = torch.device("cuda")
        for param in self.albert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.albert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
    
    
class AlbertModelTest(nn.Module):
    def __init__(self):
        super(AlbertModelTest, self).__init__()
        config = AlbertConfig.from_pretrained('models/config.json')
        self.albert = AlbertForSequenceClassification(config)  # /bert_pretrain/
        self.device = torch.device("cuda")

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.albert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
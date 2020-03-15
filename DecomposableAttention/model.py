# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:18:52 2020

@author: zhaog
"""
from torch import nn
import torch
from utils import generate_sent_masks, masked_softmax, weighted_sum
from layers import EmbedingDropout

class DecomposableAttention(nn.Module):

    def __init__(self, embeddings, f_in_dim=200, f_hid_dim=200, f_out_dim=200, 
                 dropout=0.2, embedd_dim=300, num_classes=2, device="gpu"):
        super(DecomposableAttention, self).__init__()
        self.device = device
        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.embed.float()
        self.embed.weight.requires_grad = True
        self.embed.to(device)
        #self.embeding_dropout = EmbedingDropout(p=dropout)
        self.project_embedd = nn.Linear(embedd_dim, f_in_dim)
        self.F = nn.Sequential(nn.Dropout(0.2),
                               nn.Linear(f_in_dim, f_hid_dim),
                               nn.ReLU(), 
                               nn.Dropout(0.2),
                               nn.Linear(f_hid_dim, f_out_dim), 
                               nn.ReLU())
        self.G = nn.Sequential(nn.Dropout(0.2),
                               nn.Linear(2 * f_in_dim, f_hid_dim), 
                               nn.ReLU(), 
                               nn.Dropout(0.2),
                               nn.Linear(f_hid_dim, f_out_dim), 
                               nn.ReLU())
        self.H = nn.Sequential(nn.Dropout(0.2),
                               nn.Linear(2 * f_in_dim, f_hid_dim), 
                               nn.ReLU(), 
                               nn.Dropout(0.2),
                               nn.Linear(f_hid_dim, f_out_dim), 
                               nn.ReLU())
        self.last_layer = nn.Linear(f_out_dim, num_classes)

    def forward(self, q1, q1_lengths, q2, q2_lengths):
        q1_mask = generate_sent_masks(q1, q1_lengths).to(self.device)
        q2_mask = generate_sent_masks(q2, q2_lengths).to(self.device)
        q1_embed = self.embed(q1)
        q2_embed = self.embed(q2)
        #q1_embed = self.embeding_dropout(q1_embed)
        #q2_embed = self.embeding_dropout(q2_embed)
        # project_embedd编码
        q1_encoded = self.project_embedd(q1_embed)
        q2_encoded = self.project_embedd(q2_embed)
        # Attentd
        attend_out1 = self.F(q1_encoded)
        attend_out2 = self.F(q2_encoded)
        similarity_matrix = attend_out1.bmm(attend_out2.transpose(2, 1).contiguous())
        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, q2_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), q1_mask)
        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        q1_aligned = weighted_sum(q2_encoded, prem_hyp_attn, q1_mask)
        q2_aligned = weighted_sum(q1_encoded, hyp_prem_attn, q2_mask)
        #q1_aligned = weighted_sum(attend_out2, prem_hyp_attn, q1_mask)
        #q2_aligned = weighted_sum(attend_out1, hyp_prem_attn, q2_mask)
        # compare
        compare_i = torch.cat((q1_encoded, q1_aligned), dim=2)
        compare_j = torch.cat((q2_encoded, q2_aligned), dim=2)
        v1_i = self.G(compare_i)
        v2_j = self.G(compare_j)
        # Aggregate (3.3)
        v1_sum = torch.sum(v1_i, dim=1)
        v2_sum = torch.sum(v2_j, dim=1)
        output_tolast = self.H(torch.cat((v1_sum, v2_sum), dim = 1))
        logits = self.last_layer(output_tolast)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities
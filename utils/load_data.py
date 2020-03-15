# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:26:21 2020

@author: zhaog
"""
import pandas as pd
import os
from data_utils import pad_sequences
import args
import numpy as np
import gensim
import codecs

# 加载字典
def load_vocab(vocab_file = '../data/vocab.txt'):
    path = os.path.join(os.path.dirname(__file__), vocab_file)
    vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word, vocab

# word->index
def word_index(p_sentences, h_sentences):
    word2idx, idx2word, _ = load_vocab()
    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        p_list.append(p)
        h_list.append(h)
    p_list = pad_sequences(p_list, maxlen=args.max_char_len)
    h_list = pad_sequences(h_list, maxlen=args.max_char_len)
    return p_list, h_list

# 加载word_index训练数据
def load_sentences(file, data_size=None):
    path = os.path.join(os.path.dirname(__file__), '../' + file)
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]
    p_c_index, h_c_index = word_index(p, h)
    return p_c_index, h_c_index, label

def gen_vocab(w2v_file, vocab_file):
    model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=False)
    with codecs.open(vocab_file, "w", "utf-8") as f:
        f.write(str("[PAD]") + '\r\n')
        f.write(str("[UNK]") + '\r\n')
        for word in model.index2word:
            f.write(str(word) + '\r\n')
            
def load_embeddings(embdding_path):
    word2idx, idx2word, _ = load_vocab()
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(embdding_path, binary=False)
    embedding_matrix = np.zeros((len(word2idx), w2v_model.vector_size))
    #填充向量矩阵
    for idx in idx2word:
        if idx in [0,1]:
            continue
        embedding_matrix[idx] = w2v_model[idx2word[idx]]#词向量矩阵
    return embedding_matrix

if __name__ == '__main__':
    #model = gensim.models.KeyedVectors.load_word2vec_format("../data/token_vec_300.bin", binary=False)
    #print(model.vector_size)
    pass
  

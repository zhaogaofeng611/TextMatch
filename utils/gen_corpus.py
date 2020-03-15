# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:53:13 2020

@author: zhaog
"""
import os
import pandas as pd
from hanziconv import HanziConv
import codecs
import re

''' 把句子按字分开，中文按字分，英文数字按空格, 大写转小写，繁体转简体'''
def get_word_list(query):
    query = HanziConv.toSimplified(query.strip())
    regEx = re.compile('[\\W]+')#我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')#[\u4e00-\u9fa5]中文范围
    sentences = regEx.split(query.lower())
    str_list = []
    for sentence in sentences:
        if res.split(sentence) == None:
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]

def load_data(file, data_size=None):
    path = os.path.join(os.path.dirname(__file__), '../' + file)
    df = pd.read_csv(path)
    p = df['sentence1'].values
    h = df['sentence2'].values
    return p, h

if __name__ == '__main__':
    
    sentences = []
    p_train, h_train = load_data("data/LCQMC_train.csv")
    p_dev, h_dev = load_data("data/LCQMC_dev.csv")
    p_test, h_test = load_data("data/LCQMC_test.csv")
    sentences.extend(p_train)
    sentences.extend(h_train)
    sentences.extend(p_dev)
    sentences.extend(h_dev)
    sentences.extend(p_test)
    sentences.extend(h_test)
    wikifile = codecs.open("D:\Git\wikiextractor\zhwiki_extracted\AA\wiki_00", 'r', 'utf-8')
    sen_wiki = wikifile.readlines()
    sentences.extend(sen_wiki)
    num = 0
    file = codecs.open("../data/corpus.txt", 'w', 'utf-8')
    for sentence in sentences:
        num += 1
        after = get_word_list(sentence)
        if len(after) == 0:
            continue
        file.write(" ".join(after) + '\r\n')
        if num%10000==0:
            print(str(num) + "/" + str(len(sentences)))
            file.flush
    file.close
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:56:54 2020

@author: zhaog
"""
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

class TrainWord2Vec:
    """
    训练得到一个Word2Vec模型
    """
    def __init__(self, data='../data/corpus.txt', num_features=300, min_word_count=10, context=5, incremental=False, old_path=None):
        """
        定义变量
        :param data: 用于训练的语料
        :param num_features:  返回的向量长度
        :param min_word_count:  最低词频
        :param context: 滑动窗口大小
        :param incremental: 是否进行增量训练
        :param old_path: 若进行增量训练，原始模型路径
        """
        self.data = data
        self.num_features = num_features
        self.min_word_count = min_word_count
        self.context = context
        self.incremental = incremental
        self.old_path = old_path
        
    def get_text(self):
        corpus = LineSentence(self.data)  # 加载分词语料
        return corpus

    def get_model(self, text):
        """
        从头训练word2vec模型
        :param text: 经过清洗之后的语料数据
        :return: word2vec模型
        """
        model = Word2Vec(text, size=self.num_features, min_count=self.min_word_count, window=self.context)
        return model

    def update_model(self, text):
        """
        增量训练word2vec模型
        :param text: 经过清洗之后的新的语料数据
        :return: word2vec模型
        """
        model = Word2Vec.load(self.old_path)  # 加载旧模型
        model.build_vocab(text, update=True)  # 更新词汇表
        model.train(text, total_examples=model.corpus_count, epochs=model.iter)  # epoch=iter语料库的迭代次数；（默认为5）  total_examples:句子数。
        return model

    def main(self):
        """
        主函数，保存模型
        """
        text = self.get_text()
        if self.incremental:
            model = self.update_model(text)
        else:
            model = self.get_model(text)
        # 保存模型
        model.save("../data/token_vec_300.model")
        model.wv.save_word2vec_format("../data/token_vec_300.bin", binary=False)

if __name__ == '__main__':
    trainmodel = TrainWord2Vec()
    trainmodel.main()

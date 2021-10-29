# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:12:37 2019

@author: cm
"""

import os

pwd = os.path.dirname(os.path.abspath(__file__))
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
from networks import NetworkAlbert
from classifier_utils import get_feature_test
from hyperparameters import Hyperparamters as hp


class ModelAlbertTextCNN(object):
    """
    Load NetworkAlbert TextCNN model
    """
    def __init__(self, ):
        self.albert, self.sess = self.load_model()

    @staticmethod
    def load_model():
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                albert = NetworkAlbert(is_training=False)
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                checkpoint_dir = hp.file_load_model
                print("checkpoint_dir: " + checkpoint_dir)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
        return albert, sess


MODEL = ModelAlbertTextCNN()
print("Load model finished!")


def sa(sentence):
    """
    Prediction of the sentence's sentiment.
    """
    feature = get_feature_test(sentence)
    fd = {
        MODEL.albert.input_ids: [feature[0]],
        MODEL.albert.input_masks: [feature[1]],
        MODEL.albert.segment_ids: [feature[2]],
    }
    # output = MODEL.sess.run(MODEL.albert.max_pb, feed_dict=fd)  # 输出结果和对应的概率
    # output_prob = MODEL.sess.run(MODEL.albert.preds_prob, feed_dict=fd)  # 输出概率
    # output = MODEL.sess.run(MODEL.albert.preds, feed_dict=fd)  # 输出结果
    output = MODEL.sess.run(MODEL.albert.probabilities, feed_dict=fd)  # 输出结果
    # return output_prob
    return output


if __name__ == "__main__":
    ##

    # sent = "公私募大咖集体辞职"
    # sent = "《公司业绩》倍搏集团(08331.HK)半年亏损收窄至59.1万人币"
    sent = "《公司业绩》中国再生医学(08158.HK)半年亏转盈赚1,623.9万元"
    # sent = '公私募大咖集体出手疯狂的石头遭哄抢黄金赛道倍牛股诞生高增长潜力股名单来了'
    print(sa(sent))

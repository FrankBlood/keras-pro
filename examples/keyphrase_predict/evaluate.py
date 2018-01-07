#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
evaluate
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-11-27上午9:13
@copyright: "Copyright (c) 2017 Guoxiu He. All Rights Reserved"
"""

from __future__ import print_function
from __future__ import division

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
rootdir = '/'.join(curdir.split('/')[:3])
PRO_NAME = ''
sys.path.insert(0, rootdir + '/Research/' + PRO_NAME)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

import numpy as np
from Classification.LogisticRegression import LogisticRegression
from Classification.ResidualLogisticRegression import ResidualLogisticRegression
from DataLoader import DataLoader


class Evaluator(object):
    def __init__(self, model_path):
        print("Start evaluating...")
        self.network = LogisticRegression()
        # self.network = ResidualLogisticRegression()
        self.network.build()
        self.network.load_model(model_path)

    def get_score(self, predict, target, at_n = 5):
        predict = np.array([i[0] for i in predict])
        target = np.array([i[0] for i in target])
        result = np.stack((predict, target), axis=1)
        result_at_n = result[result[:, 0].argsort()][-at_n:]
        num = np.sum(result_at_n[:, 1])
        precision = num/float(at_n)
        recall = num/np.sum(target)
        f1_score = 2*precision*recall/(precision+recall+0.0001)
        return precision, recall, f1_score

    def evaluate_one(self, input, target, at_n=5):
        predict = self.network.inference(input)
        precision, recall, f1_score = self.get_score(predict, target, at_n)
        return precision, recall, f1_score

    def evaluate_all(self, data_path, mode='train', at_n=5):
        data_loader = DataLoader()
        precision_all, recall_all, f1_score_all = [], [], []
        for feature, label in data_loader.data_generator(data_path, mode, rate=1):
            # predict = self.network.inference(feature)
            # # predict = predict.reshape(1, -1)
            # predict = np.array([i[0] for i in predict])
            # label = np.array([i[0] for i in label])
            # ll = np.stack((predict, label), axis=1)
            # ll = ll[ll[:, 0].argsort()]
            # print(ll[-10:])
            # print(np.sum(ll[-10:, 1]))
            # print(predict)
            # print(label)
            # print(np.shape(ll))
            # return
            if np.sum(label)==0:
                continue
            precision, recall, f1_score = self.evaluate_one(feature, label, at_n)
            precision_all.append(precision)
            recall_all.append(recall)
            f1_score_all.append(f1_score)
        print(len(recall_all))
        print(np.sum(precision_all), np.sum(recall_all), np.sum(f1_score_all))
        print(np.sum(precision_all)/len(precision_all),
              np.sum(recall_all)/len(recall_all),
              np.sum(f1_score_all)/len(f1_score_all))


def run_evaluate():
    # data_path = './data/inspec/'
    # data_path = './data/ke20k/'
    # data_path = './data/krapivin/'
    # data_path = './data/nus/'
    data_path = './data/semeval/'
    model_path = './models/LogisticRegression_train_Sat_Dec_30_12:44:11_2017.h5'
    # model_path = './models/LogisticRegression_inspec_train_Sat_Dec_30_15:03:30_2017.h5'
    # model_path = './models/LogisticRegression_inspec_test_Sat_Dec_30_15:06:50_2017.h5'
    # model_path = './models/LogisticRegression_krapivin_train_Sat_Dec_30_15:13:03_2017.h5'
    # model_path = './models/LogisticRegression_krapivin_test_Sat_Dec_30_15:13:44_2017.h5'
    # model_path = './models/LogisticRegression_nus_train_Sat_Dec_30_15:18:08_2017.h5'
    # model_path = './models/LogisticRegression_nus_test_Sat_Dec_30_16:16:13_2017.h5'
    # model_path = './models/LogisticRegression_semeval_train_Sat_Dec_30_16:19:36_2017.h5'
    # model_path = './models/LogisticRegression_semeval_test_Sat_Dec_30_16:20:10_2017.h5'
    mode = 'train'
    at_n = 10

    evaluator = Evaluator(model_path)
    evaluator.evaluate_all(data_path, mode, at_n)


def func():
    pass


if __name__ == "__main__":
    run_evaluate()

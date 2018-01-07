#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
train
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-11-26下午1:28
@copyright: "Copyright (c) 2017 Guoxiu He. All Rights Reserved"
"""

from __future__ import print_function
from __future__ import division

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
rootdir = '/'.join(curdir.split('/')[:3])
PRO_NAME = 'keras-pro'
sys.path.insert(0, rootdir + '/Research/' + PRO_NAME)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

import time
import numpy as np
from DataLoader import DataLoader
from Classification.LogisticRegression import LogisticRegression
from Classification.ResidualLogisticRegression import ResidualLogisticRegression

def main():
    # train_data_path = './data/inspec/'
    # test_data_path = './data/inspec/'
    # train_data_path = './data/krapivin/'
    # test_data_path = './data/krapivin/'
    # train_data_path = './data/nus/'
    # test_data_path = './data/nus/'
    # train_data_path = './data/semeval/'
    # test_data_path = './data/semeval/'
    train_data_path = './data/trainingset/'
    test_data_path = './data/ke20k/'
    mode1 = 'train'
    mode2 = 'train'

    data_loader = DataLoader()

    train_tf, train_textRank, train_pr1, train_pr2, train_label\
        = data_loader.load_from_folder(train_data_path, mode1, rate=1)
    train_new_tf, train_new_textRank, train_new_pr1, train_new_pr2, train_new_label\
        = data_loader.balance_data(train_tf, train_textRank, train_pr1, train_pr2, train_label)

    test_tf, test_textRank, test_pr1, test_pr2, test_label\
        = data_loader.load_from_folder(test_data_path, mode2, rate=1)
    test_new_tf, test_new_textRank, test_new_pr1, test_new_pr2, test_new_label\
        = data_loader.balance_data(test_tf, test_textRank, test_pr1, test_pr2, test_label)

    train_feature, train_label = data_loader.train_input(train_new_tf, train_new_textRank, train_new_pr1, train_new_pr2, train_new_label)
    test_feature, test_label = data_loader.train_input(test_new_tf, test_new_textRank, test_new_pr1, test_new_pr2, test_new_label)

    # network = LogisticRegression()
    network = ResidualLogisticRegression()
    network.build()
    network.set_optimizer(optimizer_name='nadam', lr=0.001)

    now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' '))
    bst_model_path = curdir + '/models/' + network.model_name + '_' + \
                     train_data_path.split('/')[-2] +'_' + mode1 + '_' + now_time + '.h5'

    # print(np.shape(train_feature))
    # print(np.shape(train_label))

    bst_model_path = network.train(bst_model_path, train_feature, train_label, test_feature, test_label)
    print(bst_model_path)


if __name__ == "__main__":
    main()
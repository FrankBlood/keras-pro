#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
BasicCNN
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-9-20上午9:59
@copyright: "Copyright (c) 2017 Guoxiu He. All Rights Reserved"
"""

from __future__ import print_function
from __future__ import division

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

from Network import Network

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import TimeDistributed, RepeatVector, Permute, Lambda, Bidirectional, Dropout, Flatten
from keras.layers.merge import concatenate, add, dot, multiply
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Activation
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.advanced_activations import PReLU
import numpy as np

class LogisticRegression(Network):
    def __init__(self):
        Network.__init__(self)
        self.input_dim = 3
        self.units = 1
        self.hidden_dim = 20
        self.dropout_rate = 0.5

    def build(self):
        print('Build Logistic Regression Model...')
        self.set_name("LogisticRegression")

        inputs = Input(shape=(self.input_dim,))

        preds = Dense(self.units, activation='sigmoid')(inputs)
        model = Model(inputs=inputs, outputs=preds)

        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['acc'])
        model.summary()
        self.model = model


def func():
    network = LogisticRegression()
    network.build()


if __name__ == "__main__":
    func()

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

class ResidualLogisticRegression(Network):
    def __init__(self):
        Network.__init__(self)
        self.input_dim = 3
        self.units = 1
        self.hidden_dim = 50
        self.dropout_rate = 0.5

    def resblock(self, hidden, activation):
        dense = Dense(self.hidden_dim, activation=activation)(hidden)
        dense = BatchNormalization()(dense)
        dense = Dropout(self.dropout_rate)(dense)
        cat = concatenate([dense, hidden])
        return cat

    def build(self):
        print('Build Residual Logistic Regression Model...')
        self.set_name("ResidualLogisticRegression")

        inputs = Input(shape=(self.input_dim,))

        # dense = Dense(self.hidden_dim, activation='sigmoid')(inputs)
        # dense = BatchNormalization()(dense)
        # # dense = Dropout(self.dropout_rate)(dense)
        # cat = concatenate([dense, inputs])

        res1 = inputs
        for _ in range(1):
            res1 = self.resblock(res1, activation='elu')

        res2 = inputs
        for _ in range(5):
            res2 = self.resblock(res2, activation='elu')

        res3 = inputs
        for _ in range(10):
            res3 = self.resblock(res3, activation='elu')

        cat = concatenate([res1, res2, res3, inputs])

        preds = Dense(self.units, activation='sigmoid')(cat)
        model = Model(inputs=inputs, outputs=preds)

        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['acc'])
        model.summary()
        self.model = model


def func():
    network = ResidualLogisticRegression()
    network.build()


if __name__ == "__main__":
    func()

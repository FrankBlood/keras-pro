#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
DataLoader
======

A class for something.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@time: 17-11-26下午12:42
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

import numpy as np

class DataLoader(object):
    def __init__(self):
        print("data loading...")

    def split2file(self, data_path):
        file_list = os.listdir(data_path)
        for f in file_list:

            if f == 'tf.txt':
                num = 0
                with open(data_path+f, 'r') as fp:
                    fw = open(data_path+'tf/'+str(num)+'.txt', 'w')
                    for line in fp.readlines():
                        line = line.strip()
                        if line:
                            fw.write(line+'\n')
                        else:
                            fw.close()
                            num += 1
                            fw = open(data_path + 'tf/' + str(num) + '.txt', 'w')
                    fw.close()
                    os.remove(data_path + 'tf/' + str(num) + '.txt')

            if f == 'sr.txt':
                num = 0
                with open(data_path+f, 'r') as fp:
                    fw = open(data_path + 'textRank/' + str(num) + '.txt', 'w')
                    for line in fp.readlines():
                        line = line.strip()
                        if line:
                            fw.write(line + '\n')
                        else:
                            fw.close()
                            num += 1
                            fw = open(data_path + 'textRank/' + str(num) + '.txt', 'w')
                    fw.close()
                    os.remove(data_path + 'textRank/' + str(num) + '.txt')

            if f == 'sbs.txt':
                num = 0
                with open(data_path+f, 'r') as fp:
                    fw = open(data_path + 'sbs/' + str(num) + '.txt', 'w')
                    for line in fp.readlines():
                        line = line.strip()
                        if line:
                            fw.write(line + '\n')
                        else:
                            fw.close()
                            num += 1
                            fw = open(data_path + 'sbs/' + str(num) + '.txt', 'w')
                    fw.close()
                    os.remove(data_path + 'sbs/' + str(num) + '.txt')

            if f == 'pr1.txt':
                num = 0
                with open(data_path + f, 'r') as fp:
                    fw = open(data_path + 'pr1/' + str(num) + '.txt', 'w')
                    for line in fp.readlines():
                        line = line.strip()
                        if line:
                            fw.write(line + '\n')
                        else:
                            fw.close()
                            num += 1
                            fw = open(data_path + 'pr1/' + str(num) + '.txt', 'w')
                    fw.close()
                    os.remove(data_path + 'pr1/' + str(num) + '.txt')

            if f == 'pr2.txt':
                num = 0
                with open(data_path + f, 'r') as fp:
                    fw = open(data_path + 'pr2/' + str(num) + '.txt', 'w')
                    for line in fp.readlines():
                        line = line.strip()
                        if line:
                            fw.write(line + '\n')
                        else:
                            fw.close()
                            num += 1
                            fw = open(data_path + 'pr2/' + str(num) + '.txt', 'w')
                    fw.close()
                    os.remove(data_path + 'pr2/' + str(num) + '.txt')

    def load_from_folder(self, data_path, mode='train', rate=1.0):
        tf = []
        text_rank = []
        pr1 = []
        pr2 = []
        label = []
        combine_data = {'tf': tf, 'textRank': text_rank, 'pr1': pr1, 'pr2': pr2 ,'sbs': label}
        folder_list = os.listdir(data_path)
        tmp = []
        for i in folder_list:
            if i.endswith('.txt'):
                continue
            tmp.append(i)
        folder_list = tmp
        data_length = len(os.listdir(data_path+folder_list[0]))
        Range = []
        if mode == 'train':
            print("This is train dataset.")
            Range = xrange(int(data_length*rate))
        elif mode == 'test':
            print("This is test dataset.")
            Range = xrange(int(data_length*rate), data_length)
        else:
            print("No F**K show!")
        for i in Range:
            for folder in folder_list:
                with open(data_path+folder+'/'+str(i)+'.txt', 'r') as fp:
                    for line in fp.readlines():
                        combine_data[folder].append(float(line.strip()))

        tf = combine_data['tf']
        text_rank = combine_data['textRank']
        pr1 = combine_data['pr1']
        pr2 = combine_data['pr2']
        label = combine_data['sbs']

        print("load from folder...")
        print("load", len(tf), "tf data.")
        print("load", len(text_rank), "textRank data.")
        print("load", len(pr1), "pr1 data.")
        print("load", len(pr2), "pr2 data.")
        print("load", len(label), "label.")
        print("load", np.sum(label), "postive label.")
        print("load", np.sum(label) / len(label), '% postive label.')

        return tf, text_rank, pr1, pr2, label

    def data_generator(self, data_path, mode='test', rate=1.0):
        tf = []
        text_rank = []
        pr1 = []
        pr2 = []
        label = []
        combine_data = {'tf': tf, 'textRank': text_rank, 'pr1': pr1, 'pr2': pr2, 'sbs': label}
        folder_list = os.listdir(data_path)
        tmp = []
        for i in folder_list:
            if i.endswith('.txt'):
                continue
            tmp.append(i)
        folder_list = tmp
        data_length = len(os.listdir(data_path + folder_list[0]))
        Range = []
        if mode == 'train':
            print("This is train dataset.")
            Range = xrange(int(data_length * rate))
        elif mode == 'test':
            print("This is test dataset.")
            Range = xrange(int(data_length * rate), data_length)
        else:
            print("No F**K show!")
        for i in Range:
            for folder in folder_list:
                with open(data_path + folder + '/' + str(i) + '.txt', 'r') as fp:
                    for line in fp.readlines():
                        combine_data[folder].append(float(line.strip()))

            tf = combine_data['tf']
            text_rank = combine_data['textRank']
            pr1 = combine_data['pr1']
            pr2 = combine_data['pr2']
            label = combine_data['sbs']
            yield self.train_input(tf, text_rank, pr1, pr2, label)
            tf, text_rank, pr1, pr2, label = [], [], [], [], []
            combine_data = {'tf': tf, 'textRank': text_rank, 'pr1': pr1, 'pr2': pr2, 'sbs': label}


    def load(self, data_path):
        tf = []
        textRank = []
        pr1 = []
        pr2 = []
        label = []
        file_list = os.listdir(data_path)
        for f in file_list:
            if f == 'tf.txt':
                with open(data_path+f, 'r') as fp:
                    for line in fp.readlines():
                        line = line.strip()
                        if line:
                            tf.append(float(line))
            if f == 'sr.txt':
                with open(data_path+f, 'r') as fp:
                    for line in fp.readlines():
                        line = line.strip()

                        if line:
                            textRank.append(float(line))

            if f == 'pr1.txt':
                with open(data_path+f, 'r') as fp:
                    for line in fp.readlines():
                        line = line.strip()

                        if line:
                            pr1.append(float(line))

            if f == 'pr2.txt':
                with open(data_path+f, 'r') as fp:
                    for line in fp.readlines():
                        line = line.strip()

                        if line:
                            pr2.append(float(line))

            if f == 'sbs.txt':
                with open(data_path+f, 'r') as fp:
                    for line in fp.readlines():
                        line = line.strip()
                        if line:
                            label.append(float(line))

        print("Loading...")
        print("load", len(tf), "tf data.")
        print("load", len(textRank), "textRank data.")
        print("load", len(pr1), "pr1 data.")
        print("load", len(pr2), "pr2 data.")
        print("load", len(label), "label.")
        print("load", np.sum(label), "postive label.")
        print("load", np.sum(label)/len(label), '% postive label.')

        return tf, textRank, pr1, pr2, label

    def balance_data(self, tf, textRank, pr1, pr2, label, times=10):
        new_tf, new_textRank, new_pr1, new_pr2, new_label = [], [], [], [], []
        for i, j, w, o, k in zip(tf, textRank, pr1, pr2, label):
            if k == 1:
                for _ in range(times):
                    new_tf.append(i)
                    new_textRank.append(j)
                    new_pr1.append(w)
                    new_pr2.append(o)
                    new_label.append(k)
            else:
                new_tf.append(i)
                new_textRank.append(j)
                new_pr1.append(w)
                new_pr2.append(o)
                new_label.append(k)

        print("load", len(new_tf), "tf data.")
        print("load", len(new_textRank), "textRank data.")
        print("load", len(new_pr1), "pr1 data.")
        print("load", len(new_pr2), "pr2 data.")
        print("load", len(new_label), "label.")
        print("load", np.sum(new_label), "postive label.")
        print("load", np.sum(new_label) / len(new_label), '% postive label.')

        return new_tf, new_textRank, new_pr1, new_pr2, new_label

    def train_input(self, tf, textRank, pr1, pr2, label):

        tf_array = np.array(tf).astype(np.float32)
        textRank_array = np.array(textRank).astype(np.float32)
        # pr1_array = np.array(pr1).astype(np.float32)
        pr2_array = np.array(pr2).astype(np.float32)
        label_array = (np.array(label).astype(np.float32)).reshape(len(label), 1)

        # feature = np.stack((tf_array, textRank_array, pr1_array, pr2_array), axis=-1)
        feature = np.stack((tf_array, textRank_array, pr2_array), axis=-1)
        # shuffle操作
        alldata = np.hstack((feature, label_array))
        np.random.shuffle(alldata)
        feature_rand = alldata[:, :-1]
        label_rand = (np.array(alldata[:, -1]).astype(np.float32)).reshape(len(label), 1)

        return feature_rand, label_rand


def run_load():
    data_path = './data/trainingset/'

    data_loader = DataLoader()
    tf, textRank, pr1, pr2, label = data_loader.load(data_path)
    new_tf, new_textRank, new_pr1,  new_pr2, new_label = data_loader.balance_data(tf, textRank, label)
    feature, label = data_loader.train_input(new_tf, new_textRank, new_pr1, new_pr2, new_label)
    print(np.shape(feature))
    print(np.shape(label))

def run_split2file():
    data_path = './data/semeval/'
    data_loader = DataLoader()
    data_loader.split2file(data_path)

def run_load_from_folder():
    data_path = './data/semeval/'

    data_loader = DataLoader()
    train_tf, train_textRank, train_pr1, train_pr2, train_label \
        = data_loader.load_from_folder(data_path, 'train')
    train_new_tf, train_new_textRank, train_new_pr1, train_new_pr2, train_new_label \
        = data_loader.balance_data(train_tf, train_textRank, train_pr1, train_pr2, train_label)

    # test_tf, test_textRank, test_pr1, test_pr2, test_label \
    #     = data_loader.load_from_folder(data_path, 'test')
    # test_new_tf, test_new_textRank, test_new_pr1, test_new_pr2, test_new_label \
    #     = data_loader.balance_data(test_tf, test_textRank, test_pr1, test_pr2, test_label)

def func():
    pass


if __name__ == "__main__":
    # run_load()
    run_split2file()
    # run_load_from_folder()
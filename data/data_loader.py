import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_EVnetNL(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='EVnetNL_charging_load.csv',
                 target='load', scale=True, inverse=False, timeenc=0, freq='h', cols=None, scarce=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'meta_train']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'meta_train': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.scarce = scarce
        self.__read_data__()

    def __read_data__(self):  # split training, validating, test, and meta-training data
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if self.scarce:
            df_raw = df_raw[:3000]
        total_samples = len(df_raw)
        # train : valid : test = 6 : 2 : 2 ?
        border1s = [0, int(total_samples * 0.6) - self.seq_len, int(total_samples * 0.8) - self.seq_len, 0]  # start sample
        border2s = [int(total_samples * 0.6), int(total_samples * 0.8), total_samples, total_samples]  # end sample
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:  # standard normalization
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:  # inverse normalization / data_y will not be normalized
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):  # split features and labels
        s_begin = index  # start point of input sequence
        s_end = s_begin + self.seq_len  # end point of input sequence
        r_begin = s_end - self.label_len  # start point of decoder input sequence; label_len: L_token
        r_end = r_begin + self.label_len + self.pred_len  # end point of output sequence

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]  # includes both decoder input seq and label seq
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Boulder(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Boulder_charging_load.csv',
                 target='load', scale=True, inverse=False, timeenc=0, freq='h', cols=None, scarce=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'meta_train']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'meta_train': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.scarce = scarce
        self.__read_data__()

    def __read_data__(self):  # split training, validating, test, and meta-training data
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if self.scarce:
            df_raw = df_raw[:3000]
        total_samples = len(df_raw)
        # train : valid : test = 6 : 2 : 2 ?
        border1s = [0, int(total_samples * 0.6) - self.seq_len, int(total_samples * 0.8) - self.seq_len, 0]  # start sample
        border2s = [int(total_samples * 0.6), int(total_samples * 0.8), total_samples, total_samples]  # end sample
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:  # standard normalization
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:  # data_y (label) will not be normalized
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):  # split features and labels
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PALO(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='PALO_charging_load.csv',
                 target='load', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'meta_train']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'meta_train': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):  # split training, validating, test, and meta-training data
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        total_samples = len(df_raw)
        # train : valid : test = 6 : 2 : 2 ?
        border1s = [0, int(total_samples * 0.6) - self.seq_len, int(total_samples * 0.8) - self.seq_len, 0]  # start sample
        border2s = [int(total_samples * 0.6), int(total_samples * 0.8), total_samples, total_samples]  # end sample
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:  # standard normalization
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:  # data_y (label) will not be normalized
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):  # split features and labels
        s_begin = index  # start point of input sequence
        s_end = s_begin + self.seq_len  # end point of input sequence
        r_begin = s_end - self.label_len  # start point of decoder input sequence
        r_end = r_begin + self.label_len + self.pred_len  # end point of output sequence

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]  # includes both decoder input seq and output seq
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Perth2(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Perth2_charging_load.csv',
                 target='load', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'meta_train']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'meta_train': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):  # split training, validating, test, and meta-training data
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        total_samples = len(df_raw)
        # train : valid : test = 6 : 2 : 2 ?
        border1s = [0, int(total_samples * 0.6) - self.seq_len, int(total_samples * 0.8) - self.seq_len, 0]  # start sample
        border2s = [int(total_samples * 0.6), int(total_samples * 0.8), total_samples, total_samples]  # end sample
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:  # standard normalization
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:  # inverse normalization / data_y will not be normalized
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):  # split features and labels
        s_begin = index  # start point of input sequence
        s_end = s_begin + self.seq_len  # end point of input sequence
        r_begin = s_end - self.label_len  # start point of decoder input sequence
        r_end = r_begin + self.label_len + self.pred_len  # end point of output sequence

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]  # includes both decoder input seq and output seq
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Perth1(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Perth1_charging_load.csv',
                 target='load', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'meta_train']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'meta_train': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):  # split training, validating, test, and meta-training data
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        total_samples = len(df_raw)
        # train : valid : test = 6 : 2 : 2 ?
        border1s = [0, int(total_samples * 0.6) - self.seq_len, int(total_samples * 0.8) - self.seq_len, 0]  # start sample
        border2s = [int(total_samples * 0.6), int(total_samples * 0.8), total_samples, total_samples]  # end sample
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:  # standard normalization
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:  # inverse normalization / data_y will not be normalized
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):  # split features and labels
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

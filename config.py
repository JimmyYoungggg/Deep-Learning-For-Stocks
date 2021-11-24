import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import argparse

import warnings

from glob import glob
import torch

warnings.filterwarnings("ignore")


def gen_path(path, *paths, **kw):
    store_path = os.path.join(path, *paths)
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    if 'filename' in kw:
        return os.path.join(store_path, kw['filename'])
    else:
        return store_path


class Config(object):

    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
        use_cuda = True
        self.device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

        self.max_epoch = 10
        self.ensemble_num = 5
        self.ensemble_mode = ''  # in ['normal', 'best', 'snapshot']
        self.ensemble = True if self.ensemble_mode else False

        self.train_full_model = True  # train all factor model
        self.cal_shapley = False  # whether to calculate shapley values
        self.shapley_filter_model = False  # whether to use shapley values to select best factors to train
        
        # path specification
        self.origin_data_path = '/user/uncleaneddata/'
        self.date_list_file = '/user/date_list.csv'  # specify train val test set date
        self.result_path = '/user/result/'  # result output path, should be modified
        self.result_path = gen_path(self.result_path, datetime.now().strftime("%y-%m-%d:%H-%M-%S"))
        self.date_name = datetime.strftime(datetime.today(), '%Y%m%d.%H%M')  # for naming file purpose
        
        self.all_factor_path = gen_path(self.result_path, 'all_factor_model')
        self.shapley_path = gen_path(self.result_path, 'cal_shapley')
        self.filter_factor_path = gen_path(self.result_path, 'filter_factor_model')
        self.intermediate_path = gen_path(self.origin_data_path, 'inter_file')
        self.index_dir = os.path.join(self.intermediate_path, 'datetime_index.csv')
        self.data_dir = os.path.join(self.intermediate_path, 'factor_data.dat')
        self.factor_dir = os.path.join(self.intermediate_path, 'factor_name_list.csv')
        self.index_ref_dir = os.path.join(self.intermediate_path, 'datetime_ref.csv')

        # model and architecture 
        self.network = 'LSTM'  # in ['MLP', 'CNN', 'MLPwithGAN', 'LSTM']
        self.input_size = len(pd.read_csv(self.factor_dir)["factor_name"]) - 1
        self.remain_factors = 500  # num of factors remaining after shapley filtering
        self.hidden_size_lstm = [512, 128, 32]  # hidden layer neurons
        self.hidden_size_mlp = [4096, 1024, 256]
        self.n_filter = 50  # num of convolution cores
        self.output_size = 1

        self.time_step = 1 if 'MLP' in self.network else 20  # you can change 20 to any number of time_step for lstm and cnn
        self.alpha_span = int()  # target: number of days' return, the distance between train and val set, and calculation of Sharpe
        self.val_length = int()   # days of val set  
        self.train_length = int()  # days of train set
        self.train_periods = range(0, 48)  # 48 test month, 4 years

        # hyper-parameters
        self.l2_decay = 0  # L2 regularization
        self.learning_rate = 1e-4
        self.dropout_rate = 0.6
        self.batch_size = 2048 if 'MLP' in self.network else 256
        self.batch_size_eval = self.batch_size * 4  # larger than training batch size due to less gpu memory usage
        self.min_step = 3000  # minimum training step before early stop  # todo: change from 1w2 to 1w for ensemble
        self.max_step = 15000  # maximum training step
        self.eval_interval = 100  # control the frequency of evaluation of val & test
        self.eval_train = self.eval_interval * 10  # control the frequency of evaluation of the whole train set
        self.stop_ratio = 0.8  # early stop threshold

        self.num_train_process = 5  # train 5 models the same time
        self.num_shap_process = 2  # larger gpu memory usage

        self.mode = 'all_factor_model'
        self.path = self.all_factor_path
        self.gan_loss = 0  # use gan or not, 0 is not, >0 means the weight of gan_loss in the total loss

        self.eval_and_earlystop = True 

    def all_factor_mode(self):
        self.mode = 'all_factor_model'
        self.path = self.all_factor_path
        self.hidden_size_lstm = [512, 128, 32]
        self.hidden_size_mlp = [4096, 1024, 256]
        self.batch_size_eval = self.batch_size * 4


    def cal_shap_mode(self):
        self.all_factor_mode()
        self.mode = 'cal_shapley'
        self.path = self.shapley_path

        self.batch_size = 10000  # number of samples as background example in training set
        self.batch_size_eval = 1000  # number of sample to calculate shapley values in val set

    def filter_factor_mode(self):
        self.mode = 'filter_factor_model'
        self.path = self.filter_factor_path
        self.input_size = self.remain_factors
        self.hidden_size_lstm = [512, 128, 32]
        self.hidden_size_mlp = [4096, 1024, 256]  # [2048, 512, 128]
        self.batch_size = 2048
        self.batch_size_eval = self.batch_size * 4


data_path = 'user_date_path/'


class TradingDay(object):
    """
    Create tradings from a file outside. It stores both trading days and its
    index. The date/start_date/end_date parameter's type is an integer and
    format is "%Y%m%d".

    Parameters:
    -----------
    input_file: string,
        a file contains trading calendar
    start_date/end_date: int, default None
        Load trading days from start_date and end_date. If either is None,
        load all trading days

    Attributes:
    -----------
    trading_days: numpy.array(int)
        Each element represents a trading day.
    date_map: dict(int, int)
        Key is a trading day, value is its index in trading day array.
    """

    def __init__(self, input_file=os.path.join(data_path, 'calendar.csv'),
                 start_date=None, end_date=None, area='ashare'):
        if area == 'hkshare':
            input_file = os.path.join(data_path, 'hk_calendar.csv')
        self.trading_days = \
            self._load_trading_days(input_file, start_date, end_date)
        self.date_map = \
            dict(zip(self.trading_days, np.arange(self.length)))

    @staticmethod
    def _load_trading_days(path, start_date, end_date):
        """
        Load trading days from a file.
        :param path: string, file path
        :param start_date: int
        :param end_date: int
        :return: numpy.array(int)
        """
        df = pd.read_csv(path, dtype=np.int64, names=["date"])
        trading_days = df["date"].sort_values()
        if start_date is not None or end_date is not None:
            trading_days = trading_days[
                (trading_days >= start_date) & (trading_days <= end_date)]
        return trading_days.values

    @property
    def start_date(self):
        return self.trading_days[0]

    @property
    def end_date(self):
        return self.trading_days[-1]

    @property
    def length(self):
        return len(self.trading_days)

    def distance(self, start_date, end_date):
        """Get distance from start_date and end_date"""
        return self.date_map[end_date] - self.date_map[start_date]

    def get_loc(self, date_):
        """Get a date's index in trading days"""
        return self.date_map[date_]

    def is_trading_day(self, date_):
        try:
            num = self.date_map[date_]
            return True
        except KeyError:
            return False

    def last_trading_day(self, date_):
        """Get last trading dat"""
        return self.trading_days[self.trading_days < date_][-1]

    def next_trading_day(self, date_):
        """Get next trading day of date"""
        return self.trading_days[self.trading_days > date_][0]

    def day_after_last(self, date_):
        """ Get the day after last trading day, it is always equal to the
        input date when input date is not Friday.
        """
        last_trading_day = self.last_trading_day(date_)
        datetime_ = datetime.strptime(str(last_trading_day), "%Y%m%d")
        datetime_ += timedelta(days=1)
        return int(datetime_.strftime("%Y%m%d"))

    def trading_day_pair(self, date_):
        """Get the minimum interval of trading days contains date"""
        if self.is_trading_day(date_):
            return date_, self.next_trading_day(date_)
        else:
            return self.last_trading_day(date_), self.next_trading_day(date_)

    def get_range(self, start_date, end_date):
        return self.trading_days[(self.trading_days >= start_date) &
                                 (self.trading_days <= end_date)]


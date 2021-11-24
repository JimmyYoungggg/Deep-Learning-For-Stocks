import numpy as np
import pandas as pd
import os
import torch
import re

from torch.utils.data import DataLoader, Dataset
from config import TradingDay, gen_path
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
from math import pi, cos


def load_dataset(start_date, end_date, config):
    """
    Read a dataset from local file given its start_date and end_date
    :return: arr: np.array, (sample_size, factor_size);
             trade_date, np.array, (sample_size,);
             symbol, np.array, (sample_size,)
    """
    datetime_index = pd.read_csv(config.index_dir)
    datetime_ref = pd.read_csv(config.index_ref_dir, index_col=0, dtype={"symbol": "str"})
    datetime_index["date"] = pd.to_datetime(datetime_index['datetime']).dt.date
    start_index = np.where(datetime_index["date"].values >=
                           pd.to_datetime(str(start_date), format="%Y-%m-%d").date())[0][0]
    end_index = np.where(datetime_index["date"].values <
                         pd.to_datetime(str(end_date), format="%Y-%m-%d").date())[0][-1]
    datetime_index = datetime_index.iloc[start_index:end_index + 1]  # select dates needed here
    trade_date = datetime_index["date"].values
    valid_datetime = datetime_index["datetime"].values
    symbol = datetime_ref.loc[valid_datetime, "symbol"].values

    factor_cols = pd.read_csv(config.factor_dir)["factor_name"]
    factor_size = len(factor_cols)
    offset = start_index
    count = end_index - start_index + 1
    with open(config.data_dir, "rb") as afile:
        afile.seek(offset * factor_size * 4)
        arr = np.fromfile(afile, count=count * factor_size, dtype=np.float32)
    arr = arr.reshape(-1, factor_size)
    return arr, trade_date, symbol


def get_index_matrix(dataset, time_step):
    """
    Link a symbol's some trade date to its previous time_step dates, mainly serve for LSTM model which needs time series
    data to make predictions, but is also compatible with MLP models setting time_step to 1.
    """
    unique_symbol = np.unique(dataset[2])
    index_matrix = np.zeros((dataset[0].shape[0], time_step), dtype=int)
    valid_index = []
    for symbol in unique_symbol:
        index_symbol = np.where(dataset[2] == symbol)[0]  # all the index for one symbol in the data
        if len(index_symbol) < time_step:
            continue
        for j in range(time_step, len(index_symbol) + 1):
            # for each symbol, the first (time_step-1) y data would be abandoned
            # first (time_step-1) features data will be retained as part of the input of the first sample of that symbol
            index_matrix[index_symbol[j - 1], :] = index_symbol[j - time_step:j]
            valid_index.append(index_symbol[j - 1])

    # ----index_matrix shape(data[0].shape[0], time_step), for each sample(i.e. a symbol on one day),
    # the columns are the index of its previous time_step days
    # ----valid_index: for each symbol, the first time_step-1 samples would be invalid, but for each data set,
    # the start date has been pushed (time_step-1) forward to ensure the original start date is a valid sample;
    # and all the symbols whose number of trading days in this period is less that time_step would be invalid.
    return index_matrix, valid_index


class Mydataset(Dataset):
    """
    construct a mapable data_set from raw data
    help link an index to a sample, which is what __getitem__(item) does
    to be passed to dataloader
    """
    def __init__(self, data, valid_index, index_matrix, factor_num, config, period):
        self.data = data
        self.valid_index = valid_index
        self.index_matrix = index_matrix
        self.factor_num = factor_num
        self.config = config
        self.period = period
        if config.mode == 'filter_factor_model':
            self.shap_x_index = np.loadtxt(gen_path(config.shapley_path, filename=str(period)+'.txt')).astype(int)

    def __getitem__(self, item):
        matrix_row = self.valid_index[item]  # matrix_row is the valid indice for index_metrix and data_date, symbol
        keys = self.index_matrix[matrix_row]  # keys should be 5 index (if time_step == 5) of a symbol's previous 5 days
        sample = self.data[keys, :]
        if self.config.network == 'TransAm':
            X = sample[:-1, :]
            y = sample[-1, -1]
        else:
            X = sample[:, :self.factor_num]
            y = sample[-1, self.factor_num:self.factor_num + 1]
        if self.config.mode == 'filter_factor_model':
            X = X[:, self.shap_x_index[:self.config.input_size]]
        return X, y, matrix_row

    def __len__(self):
        return len(self.valid_index)  # near 3 million in train_set, 180k in val_set, 60k in test_set


def generate_dataloader(config, period, val_length, train_length):
    print('creating dataloader')
    with open(gen_path(config.path, str(period), filename='log.txt'), 'a') as logfile:
        data_dict = {}
        factor_series = pd.read_csv(config.factor_dir)["factor_name"]
        factor_num = len(factor_series) - 1  # the last factor is the label
        s = factor_series.map(lambda x: x[:3])
        env_index = np.where(s == 'env')[0]  # locate factors that start with env
        de_extreme = list(set(range(factor_num)) - set(env_index))
        #  determine start & end date for each set based on a local file
        '''Determine original start & end date for each data set first'''
        CALENDER = TradingDay()
        date_list = pd.read_csv(config.date_list_file)
        test_start_date, test_end_date = date_list['testing_start_date'][period], date_list['testing_end_date'][period]
        # val set should be 10 trading days earlier than test, train set should be 10 trading days earlier than val
        val_end_date = CALENDER.trading_days[CALENDER.get_loc(test_start_date) - config.alpha_span]
        val_start_date = CALENDER.trading_days[CALENDER.get_loc(val_end_date) - val_length]
        train_end_date = CALENDER.trading_days[CALENDER.get_loc(val_start_date) - max(config.alpha_span,
                                                                                      config.time_step)]
        train_start_date = CALENDER.trading_days[CALENDER.get_loc(train_end_date) - train_length]

        logfile.write(''.join(
         ['Generating dataloader for training set, period: ', str(train_start_date), ' to ', str(train_end_date),
          '\n']))
        # Start_date of each set is pushed (time_step-1) days forward to ensure original start date is a valid sample
        train_start_date = CALENDER.trading_days[CALENDER.get_loc(train_start_date) - config.time_step + 1]
        data_date = load_dataset(train_start_date, train_end_date, config)
        # prepare index for selecting samples, which are to be passed to Mydataset
        index_matrix, valid_index = get_index_matrix(data_date, config.time_step)
        X_data, y_data = data_date[0][:, :factor_num], data_date[0][:, factor_num:factor_num + 1]
        low_bar = np.percentile(X_data[:, de_extreme], 1, axis=0, keepdims=True)
        high_bar = np.percentile(X_data[:, de_extreme], 99, axis=0, keepdims=True)
        np.save(gen_path(config.path, str(period), 'scaler', filename='low_bar.npy'), low_bar)
        np.save(gen_path(config.path, str(period), 'scaler', filename='high_bar.npy'), high_bar)
        X_data[:, de_extreme] = np.clip(X_data[:, de_extreme], low_bar, high_bar)  # strip X of extremes.
        y_data = np.clip(y_data, -0.3, 0.3)
        sc_X = StandardScaler().fit(np.array(X_data))
        sc_y = StandardScaler().fit(np.array(y_data))
        joblib.dump(sc_X, gen_path(config.path, str(period), 'scaler', filename='training_sc_X.pkl'))
        joblib.dump(sc_y, gen_path(config.path, str(period), 'scaler', filename='training_sc_y.pkl'))
        X_tsf = sc_X.transform(X_data)
        y_tsf = sc_y.transform(y_data)
        _data = np.concatenate((X_tsf, y_tsf), axis=1)
        _set = Mydataset(_data, valid_index, index_matrix, factor_num, config, period)
        dataloader = DataLoader(_set, batch_size=config.batch_size, shuffle=True, drop_last=True)
        data_dict['train'] = [dataloader, data_date[1], data_date[2]]

        logfile.write(''.join(
         ['Generating dataloader for validation set, period: ', str(val_start_date), ' to ', str(val_end_date),
          '\n']))
        val_start_date = CALENDER.trading_days[CALENDER.get_loc(val_start_date) - config.time_step + 1]
        data_date = load_dataset(val_start_date, val_end_date, config)
        index_matrix, valid_index = get_index_matrix(data_date, config.time_step)
        X_data, y_data = data_date[0][:, :factor_num], data_date[0][:, factor_num:factor_num + 1]
        X_data[:, de_extreme] = np.clip(X_data[:, de_extreme], low_bar, high_bar)  # use train stat to strip extremes
        X_tsf = sc_X.transform(X_data)
        y_tsf = sc_y.transform(y_data)  # use train stat to standardize
        _data = np.concatenate((X_tsf, y_tsf), axis=1)
        _set = Mydataset(_data, valid_index, index_matrix, factor_num, config, period)
        dataloader = DataLoader(_set, batch_size=config.batch_size_eval, shuffle=True, drop_last=False)
        data_dict['val'] = [dataloader, data_date[1], data_date[2]]

        logfile.write(''.join(
         ['Generating dataloader for test set, period: ', str(test_start_date), ' to ', str(test_end_date), '\n']))
        test_start_date = CALENDER.trading_days[CALENDER.get_loc(test_start_date) - config.time_step + 1]
        data_date = load_dataset(test_start_date, test_end_date, config)
        index_matrix, valid_index = get_index_matrix(data_date, config.time_step)
        X_data, y_data = data_date[0][:, :factor_num], data_date[0][:, factor_num:factor_num + 1]
        X_data[:, de_extreme] = np.clip(X_data[:, de_extreme], low_bar, high_bar)  # use train stat to strip extremes
        X_tsf = sc_X.transform(X_data)
        y_tsf = sc_y.transform(y_data)  # use train stat to standardize
        _data = np.concatenate((X_tsf, y_tsf), axis=1)
        _set = Mydataset(_data, valid_index, index_matrix, factor_num, config, period)
        # data_dict['test'] = [_set, data_date[1], data_date[2]]
        dataloader = DataLoader(_set, batch_size=config.batch_size_eval, shuffle=True, drop_last=False)
        data_dict['test'] = [dataloader, data_date[1], data_date[2]]

    return data_dict


def process_origin_data(config):
    """
    Clean origin data, drop nan, filter samples with universe, archive data by date order instead of factor order
    """
    data_dir, intermediate_path = config.origin_data_path, config.intermediate_path
    origin_data_time = os.path.getmtime(os.path.join(data_dir, "factor_data.dat"))
    clean_data_path = os.path.join(intermediate_path, "factor_data.dat")

    if os.path.exists(clean_data_path):
        if origin_data_time > os.path.getmtime(clean_data_path):
            print('Processed data are obsolete, deleting them')
            del_list = os.listdir(intermediate_path)
            for f in del_list:
                file_path = os.path.join(intermediate_path, f)
                os.remove(file_path)
    if not os.path.exists(os.path.join(config.intermediate_path, 'datetime_ref.csv')):
        print('Processed data not existed, processing origin data')
        drop_label_names = ['ret_origin_window_10', 'ret_alpha_window_10', 'ret_standard_window_10']
        drop_label_names.extend(['ret_origin_window_15', 'ret_alpha_window_15', 'ret_standard_window_15'])
        drop_label_names.extend(['ret_origin_window_20', 'ret_alpha_window_20', 'ret_standard_window_20'])
        drop_label_names.append('universe')
        label_name = 'ret_alpha_window_10'
        drop_label_names.remove(label_name)

        columns = pd.read_csv(os.path.join(data_dir, "factor_name_all.csv"))
        columns_all = columns["factor_name"].values.reshape([-1, ])
        indices = pd.read_csv(os.path.join(data_dir, "datetime.csv"))
        indices = pd.to_datetime(indices["datetime"])
        date_time_all = indices.values.reshape([-1, ])

        num_factor = len(columns_all)  # 1274 include universe
        num_sample = len(date_time_all)  # 12995140

        num_chunk = 3
        chunk_load = num_sample // num_chunk

        data = np.fromfile(os.path.join(data_dir, "factor_data.dat"))
        data = data.reshape(num_factor, num_sample).T

        for i in range(num_chunk):
            if i == num_chunk - 1:  # the last chunk might be larger
                arr = data[i * chunk_load:]
            else:
                arr = data[i * chunk_load: (i + 1) * chunk_load]
            file_path = gen_path(intermediate_path, filename=str(i).join(['samples', '.npy']))
            np.save(file_path, arr)
        del arr, data

        valid_rows = np.zeros(num_sample, dtype=np.bool)
        uni_col = np.where(columns_all == "universe")[0][0]
        npy_file_list = sorted(os.listdir(intermediate_path))
        with open(os.path.join(intermediate_path, 'factor_data.dat'), 'wb') as _file:
            for i, npy_file in enumerate(npy_file_list):
                data_np = np.load(gen_path(intermediate_path, filename=npy_file))
                data_np_uni = data_np[:, uni_col]
                na_cond_col = (np.isnan(data_np)).sum(axis=0) < data_np.shape[0]
                nan_factors = np.where(na_cond_col == False)[0]
                assert len(nan_factors) == 0, 'Factor {} all nan'.format(columns_all[nan_factors])
                na_cond_col = np.logical_and(na_cond_col, [each not in drop_label_names for each in columns_all])
                choose_cols = np.where(na_cond_col)[0]
                data_np = data_np[:, choose_cols]
                choose_rows = np.where(((np.isnan(data_np)).sum(axis=1) == 0) & (data_np_uni != 0.))[0]
                data_np = data_np[choose_rows, :]
                choose_rows += i * chunk_load
                valid_rows[choose_rows] = True
                data_np[:, -1] = np.clip(data_np[:, -1], -0.3, 0.3)
                _file.write(data_np.reshape(-1, ).astype(np.float32))
                os.remove(gen_path(intermediate_path, filename=npy_file))
                del data_np

        factor_name = pd.DataFrame({'factor_name': columns_all[choose_cols]})
        datetime_index = pd.DataFrame({'datetime': date_time_all[valid_rows]})
        factor_name.to_csv(os.path.join(intermediate_path, 'factor_name_list.csv'), index=False)
        datetime_index.to_csv(os.path.join(intermediate_path, 'datetime_index.csv'), index=False)
        all_datetime = pd.read_csv(os.path.join(data_dir, 'datetime.csv'))
        all_symbol = pd.read_csv(os.path.join(data_dir, 'symbol.csv'), dtype={'symbol': str})
        all_date = pd.read_csv(os.path.join(data_dir, 'date.csv'))
        datetime_ref = pd.concat([all_datetime, all_symbol, all_date], axis=1, sort=False)
        datetime_ref.to_csv(os.path.join(intermediate_path, 'datetime_ref.csv'), index=False)


def snapshot_lr(initial_lr, iteration, epoch_per_cycle):
    # proposed CosineAnnealing learning late function
    return initial_lr * (cos(pi * (iteration % epoch_per_cycle)) + 1) / 2



def make_prediction(dataloader, sc_y, model, config):
    """
    Make predictions about a whole dataset based on a given model
    :param dataloader: pytorch dataloader of a dataset
    :param sc_y: statistics values object to inverse-transform prediction
    :param model: a given model, or a list of model in the case of ensemble
    :param config:
    :return: ground truth, prediction, indexing
    """
    torch.cuda.empty_cache()  # only to be monitored by nvidia-smi
    if isinstance(model, list):  # in the case of ensemble
        for m in model:
            m.eval()
    else:
        model.eval()
    with torch.no_grad():
        pred_y_list, y_list, valid_index_list = [], [], []
        data_iterator = iter(dataloader)
        '''Load the whole validation set'''
        while True:
            try:
                (X_batch, y_batch, valid_index) = next(data_iterator)
            except StopIteration:
                pred_y, y = np.concatenate(pred_y_list, axis=0), np.concatenate(y_list, axis=0)
                valid_index_all = np.concatenate(valid_index_list, axis=0)
                break
            else:
                X_batch = torch.Tensor(X_batch).to(config.device)
                if config.ensemble:
                    pred_y_batch = []
                    for m in model:
                        pred_y_batch_tmp = m(X_batch).cpu().numpy()
                        pred_y_batch.append(pred_y_batch_tmp)
                    pred_y_batch = np.array(pred_y_batch).mean(axis=0)
                else:
                    pred_y_batch = model(X_batch)
                    pred_y_batch = pred_y_batch.cpu().numpy()
                pred_y_list.append(pred_y_batch)
                y_list.append(y_batch)
                valid_index_list.append(valid_index)

    '''Inverse-transform prediction and target before calculate performance and comparable mse loss'''
    predict_y_inv = sc_y.inverse_transform(pred_y.squeeze())
    real_y_inv = sc_y.inverse_transform(y.squeeze())
    return predict_y_inv, real_y_inv, valid_index_all


def save_model_prediction(model, period, current_round, set_data, sc_y, config, set_name):
    """
    Save model_prediction on a dataset, analysis on training process could be done on these prediction
    """
    dataloader_test = set_data[0]
    test_date = set_data[1]
    test_symbol = set_data[2]
    predict_y_test, real_y_test, valid_index_test = make_prediction(dataloader_test, sc_y, model, config)

    stock_score = pd.DataFrame()
    stock_score["symbol"] = test_symbol[valid_index_test]
    stock_score["score"] = predict_y_test
    stock_score['truth'] = real_y_test
    stock_score["date"] = test_date[valid_index_test]
    stock_score = stock_score.sort_values(by=["date"])
    stock_score.to_hdf(gen_path(config.path, set_name+'_scores', str(period), filename=str(current_round) + '.h5'), key='df')


def cal_performance(predict_y_inv, real_y_inv, _date, valid_index_all, set_name, writer, current_round, config):
    """
    Calculate some performance metrics of a given dataset's prediction and ground truth, and draw them on tensorboard
    :param predict_y_inv: prediction
    :param real_y_inv: ground truth
    """
    result_df = pd.DataFrame()
    result_df["y_pred"] = predict_y_inv
    result_df["y"] = real_y_inv
    coef_all = result_df['y_pred'].corr(result_df['y'])
    result_df["date"] = _date[valid_index_all]
    grouped = result_df.groupby("date")
    result_df['symbols_num'] = grouped['date'].transform('count')
    result_df['y_rank'] = grouped['y'].transform(lambda x: x.rank(ascending=False))
    result_df['accurate_head'] = (result_df['y_rank'] <= 250).map(lambda x: int(x))  # 1 if its intraday rank <=250
    result_df['accurate_tail'] = (result_df['y_rank'] > result_df['symbols_num'] - 250).map(lambda x: int(x))
    alpha_daily_all = grouped['y'].mean()
    alpha_all_mean, alpha_all_std = np.average(alpha_daily_all.values), np.std(alpha_daily_all.values)
    sharpe_ratio_all = alpha_all_mean / alpha_all_std * np.sqrt(250 / config.alpha_span)
    '''calculation in group_head'''
    df_group_head = grouped.apply(lambda x: x.sort_values(['y_pred'], ascending=False).head(250))
    df_group_head.index = df_group_head.index.droplevel(0)
    accuracy_head = df_group_head.groupby("date")['accurate_head'].mean().values  # it is a array with length of dates
    alpha_daily_head = df_group_head.groupby("date")["y"].mean()
    alpha_head_mean, alpha_head_std = np.average(alpha_daily_head.values), np.std(alpha_daily_head.values)
    sharpe_ratio_head = alpha_head_mean / alpha_head_std * np.sqrt(250 / config.alpha_span)
    '''calculation in group_tail'''
    df_group_tail = grouped.apply(lambda x: x.sort_values(['y_pred'], ascending=False).tail(250))
    df_group_tail.index = df_group_tail.index.droplevel(0)
    accuracy_tail = df_group_tail.groupby("date")['accurate_tail'].mean().values
    alpha_daily_tail = df_group_tail.groupby("date")["y"].mean()
    alpha_tail_mean, alpha_tail_std = np.average(alpha_daily_tail.values), np.std(alpha_daily_tail.values)
    sharpe_ratio_tail = alpha_tail_mean / alpha_tail_std * np.sqrt(250 / config.alpha_span)

    Mse = torch.nn.MSELoss(reduction='mean')  # i.e. Mean square error, return an average error overall
    mse_y = Mse(torch.from_numpy(predict_y_inv), torch.from_numpy(real_y_inv))

    if writer:
        '''Document performance in tensorboard'''
        writer.add_scalars('MSE', {set_name: mse_y}, current_round)
        writer.add_scalars('corr_all', {set_name: coef_all}, current_round)
        writer.add_scalars(set_name + '_sharpe',
                           {'all': sharpe_ratio_all, 'top': sharpe_ratio_head, 'tail': sharpe_ratio_tail}, current_round)
        writer.add_scalars(set_name + '_alpha_mean',
                           {'all': alpha_all_mean, 'top': alpha_head_mean, 'tail': alpha_tail_mean}, current_round)
        writer.add_scalars(set_name + '_alpha_std',
                           {'all': alpha_all_std, 'top': alpha_head_std, 'tail': alpha_tail_std}, current_round)
        writer.add_scalars(set_name + '_accuracy',
                           {'top': np.average(accuracy_head), 'tail': np.average(accuracy_tail)}, current_round)
        writer.add_histogram(set_name + '_real_y', result_df['y'].values, current_round, bins=50)
        writer.add_histogram(set_name + '_pred_y', result_df['y_pred'].values, current_round, bins=50)

    dic_result = {
        'MSE': mse_y,
        'corr_all': coef_all,
        'sharpe': {'all': sharpe_ratio_all, 'top': sharpe_ratio_head, 'tail': sharpe_ratio_tail},
        'alpha_mean': {'all': alpha_all_mean, 'top': alpha_head_mean, 'tail': alpha_tail_mean},
        'alpha_std': {'all': alpha_all_std, 'top': alpha_head_std, 'tail': alpha_tail_std},
        'accuracy': {'top': np.average(accuracy_head), 'tail': np.average(accuracy_tail)},
    }
    return sharpe_ratio_head, alpha_head_mean, dic_result


def process_stock_score(path, all_periods):
    """
    Only concat separate period prediction to a single csv for backtest purpose
    """
    frames = []
    for i in all_periods:
        df = pd.read_csv(gen_path(path, 'stock_score', filename=str(i) + '.csv'))
        df['symbol'] = df['symbol'].map(lambda x: str(x))
        df['symbol'] = df['symbol'].map(lambda x: str(0) * (6 - len(x)) + x if len(x) < 6 else x)
        df['date'] = df['date'].map(lambda x: int(x[:4] + x[5:7] + x[-2:]))
        df['score'] = df['score'].map(lambda x: round(x, 10))
        frames.append(df)
    result = pd.concat(frames)
    result.index = result['symbol']
    del result['symbol']
    result.to_csv(gen_path(path, filename='all_stock_score.csv'))


def process_resulttxt(path):

    filename = gen_path(path, filename='result.txt')
    periods, val_sharpe, test_sharpe, test_alpha = list(), list(), list(), list()
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            m = re.match('Period (.*), final round (.*) and its val sharpe (.*) & test sharpe (.*) & test alpha (.*)', lines)
            periods.append(int(m.group(1)))
            val_sharpe.append(float(m.group(3)))
            test_sharpe.append(float(m.group(4)))
            test_alpha.append(float(m.group(5)))
    result = pd.DataFrame({'periods': periods, 'val_sharpe': val_sharpe,
                           'test_sharpe': test_sharpe, 'test_alpha': test_alpha})
    result = result[['periods', 'val_sharpe', 'test_sharpe', 'test_alpha']]
    result.set_index('periods', inplace=True)
    result.sort_index(inplace=True)
    result.to_csv(gen_path(path, filename='result.csv'))
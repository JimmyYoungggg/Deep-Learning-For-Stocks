import numpy as np
import pandas as pd
import os
import torch
import shap
import multiprocessing as mp
from functools import partial
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from config import Config, gen_path
from models import MLP, MLPwithGAN, VanillaLSTM, GAN, CNNfeature
from utils import *
import copy
from glob import glob

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(1)
np.random.seed(1)


def train_model(config, period, train_data, val_data, test_data):
    '''Training and evluation during training'''
    '''Prepare data, tensorboard, model, optimizer, loss_fn, standard_scalar, plt figure and so on for training'''
    dataloader_train = train_data[0]
    train_date = train_data[1]
    dataloader_val = val_data[0]
    val_date = val_data[1]
    dataloader_test = test_data[0]
    test_date = test_data[1]
    writer = SummaryWriter(gen_path(config.path, str(period), 'tensorboard'))
    writer.add_text(config.network, config.date_name + "input_size={:d}, time_step={:d}, batch_size={:d}, "
                                                       "hidden_size_lstm={}, hidden_size_mlp={}, dropout_rate={:.1f}, "
                                                       "learning_rate={:.6f}, l2_decay={:5f}, time_period={:d}\n".format(
        config.input_size, config.time_step, config.batch_size, config.hidden_size_lstm, config.hidden_size_mlp,
        config.dropout_rate, config.learning_rate, config.l2_decay, period))
    if config.network == 'MLPwithGAN':
        model = MLPwithGAN(config)
    elif config.network == 'MLP':
        model = MLP(config)
    elif config.network == 'LSTM':
        model = VanillaLSTM(config)
    elif config.network == 'CNN':
        model = CNNfeature(config)
    else:
        raise Exception('Unknown model type:{}'.format(config.network))
    print(model)
    model.to(config.device)
    print('start training ', period, 'model on', next(model.parameters()).is_cuda, 'tensor to', config.device)

    if config.ensemble:  
        snapshot_model = []

    if config.gan_loss > 0:
        gan = GAN(config)
        gan.to(config.config.device)
        gan.train()
        optimizer_gan = torch.optim.Adam(gan.parameters(), lr=config.learning_rate * 1e-1, weight_decay=config.l2_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_decay)

    Mse = torch.nn.MSELoss(reduction='mean')

    sc_y = joblib.load(gen_path(config.path, str(period), 'scaler', filename='training_sc_y.pkl'))

    valid_sharpe_max = float("-inf")
    model_renewal = []

    train_iterator = iter(dataloader_train)
    val_iterator = iter(dataloader_val)

    for current_round in range(int(config.max_step * 256 / config.batch_size)):

        if config.ensemble and config.ensemble_mode == 'snapshot':
            lr = snapshot_lr(config.learning_rate, current_round, int(config.max_step * 256 / (config.batch_size * config.ensemble_num)))
            optimizer.state_dict()["param_groups"][0]["lr"] = lr

        '''Begin training'''
        model.train()
        optimizer.zero_grad()

        '''Fetch a training batch'''
        try:
            (_train_X, _train_y, _train_valid_index) = next(train_iterator)
            _train_X, _train_y = _train_X.to(config.device), _train_y.to(config.device)
        except StopIteration:
            train_iterator = iter(dataloader_train)
            (_train_X, _train_y, _train_valid_index) = next(train_iterator)
            _train_X, _train_y = _train_X.to(config.device), _train_y.to(config.device)

        try:
            (_val_X, _val_y, _val_valid_index) = next(val_iterator)
            _val_X = _val_X.to(config.device)  # requires_grad=False
        except StopIteration:
            val_iterator = iter(dataloader_val)
            (_val_X, _val_y, _val_valid_index) = next(val_iterator)
            _val_X = _val_X.to(config.device)

        '''Make predictions according to the model'''
        if config.network in ['MLP', 'LSTM', 'CNN']:
            pred_y_train = model(_train_X)
        elif config.network == 'MLPwithGAN':
            pred_y_train, representation = model(_train_X, True)
            _, representation_val = model(_val_X, True)

        '''Calculate loss'''
        pred_loss = Mse(pred_y_train, _train_y)   # Mse(pred_y_train, _train_y)
        if config.gan_loss > 0:
            pred_domain = gan(representation)
            pred_domain_val = gan(representation_val)
            '''GAN Loss for the generator, train the generator'''
            gan_loss = (Mse(pred_domain, torch.ones(pred_domain.shape).to(config.device) / 2) + \
                        Mse(pred_domain_val, torch.ones(pred_domain_val.shape).to(config.device) / 2)) / 2
        else:
            gan_loss = 0
        loss = pred_loss + config.gan_loss * gan_loss

        if current_round % config.eval_train == 1:
            print('period: {:d}\tround: {:d}\tloss: {:.4f}\tpred_loss: {:.4f}\tgan_loss: {:.4f}\tinfo_loss: {:.4f}\t'
                  'orth_loss: {:.4f}'.format(period, current_round, loss, pred_loss, gan_loss, info_loss, orth_loss))

        '''Calculate gradients for all tensors that require grad, i.e. parameters'''
        loss.backward()

        '''Update all tensors that require grad, i.e. parameters'''
        optimizer.step()

        if config.gan_loss > 0:
            gan.train()
            optimizer_gan.zero_grad()

            _, representation = model(_train_X, True)
            _, representation_val = model(_val_X, True)

            pred_domain = gan(representation)
            pred_domain_val = gan(representation_val)

            '''GAN Loss for the discriminator, train the discriminator'''
            gan_loss = Mse(pred_domain, torch.ones(pred_domain.shape).to(config.device)) + \
                       Mse(pred_domain_val, torch.zeros(pred_domain_val.shape).to(config.device))
            gan_loss.backward()
            optimizer_gan.step()

        '''Evaluate the whole train set during training'''
        if current_round > int(config.min_step * 256 / config.batch_size) and \
                current_round % int(config.eval_train * 256 / config.batch_size) == 1 and config.eval_and_earlystop:  # todo: ban this now
            predict_y_train, real_y_train, valid_index_train = make_prediction(dataloader_train, sc_y, model, config)
            _, _, dic_re = cal_performance(predict_y_train, real_y_train, train_date, valid_index_train, 'train',
                                           writer,
                                           current_round, config)

        if config.ensemble and config.ensemble_mode == 'normal':
            temp = copy.deepcopy(model)
            snapshot_model.append(temp.cpu())
            snapshot_model = snapshot_model[-config.ensemble_num:]
        elif config.ensemble and config.ensemble_mode == 'snapshot':
            if current_round % int(config.max_step * 256 / (config.batch_size * config.ensemble_num)) - \
                    int(config.max_step * 256 / (config.batch_size * config.ensemble_num)) == -1:
                temp = copy.deepcopy(model)
                snapshot_model.append(temp.cpu())
                snapshot_model = snapshot_model[-config.ensemble_num:]

        '''Evaluate val & test set during training'''
        if current_round > int(config.min_step * 256 / config.batch_size) and \
                current_round % int(config.eval_interval * 256 / config.batch_size) == 1 and config.eval_and_earlystop:  # todo: ban this now
            predict_y_val, real_y_val, valid_index_val = make_prediction(dataloader_val, sc_y, model, config)
            sharpe_head_val, alpha_head_val, dic_re = cal_performance(predict_y_val, real_y_val, val_date,
                                                                      valid_index_val,
                                                                      'val', writer, current_round, config)

            predict_y_test, real_y_test, valid_index_test = make_prediction(dataloader_test, sc_y, model, config)
            sharpe_head_test, alpha_head_test, dic_re = cal_performance(predict_y_test, real_y_test, test_date,
                                                                        valid_index_test,
                                                                        'test', writer, current_round, config)
            for name, para in model.named_parameters():  # draw model parameters' values and grads
                if para.grad is None:
                    print(name + ':grad is None')
                else:
                    writer.add_histogram(name + '_grad', para.grad.cpu().data.numpy(), current_round, bins=50)
                    writer.add_histogram(name + '_data', para.cpu().data.numpy(), current_round, bins=50)

            if sharpe_head_val > valid_sharpe_max:
                if config.ensemble and config.ensemble_mode == 'best':
                    temp = copy.deepcopy(model)
                    snapshot_model.append(temp.cpu())
                    snapshot_model = snapshot_model[-config.ensemble_num:]

            '''Early stop setting'''
            if current_round > int(config.min_step * 256 / config.batch_size) and config.eval_and_earlystop:
                if sharpe_head_val > valid_sharpe_max:
                    valid_sharpe_max = sharpe_head_val
                    model_renewal.append((current_round, sharpe_head_val, sharpe_head_test, alpha_head_test))
                    with open(gen_path(config.path, str(period), filename='train_log.txt'), 'a') as logfile:
                        logfile.write('Model renewal at round {:d} and its val sharpe {:.2f} & test sharpe {:.2f} & '
                                      'test alpha {:.4f} \n'.format(current_round, sharpe_head_val, sharpe_head_test,
                                                                    alpha_head_test))
                    torch.save(model.state_dict(),
                               gen_path(config.path, str(period), 'model', filename=config.network + '.pkl'))
                else:
                    if valid_sharpe_max >= 0:
                        stop_threshold = config.stop_ratio * valid_sharpe_max
                    else:
                        stop_threshold = (1. / config.stop_ratio) * valid_sharpe_max
                    if sharpe_head_val <= stop_threshold:  # early_stop
                        break

    if config.ensemble:
        if config.ensemble_mode == 'snapshot':
            if len(snapshot_model) < config.ensemble_num:
                temp = copy.deepcopy(model)
                snapshot_model.append(temp.cpu())
                snapshot_model = snapshot_model[-config.ensemble_num:]

        for i in range(len(snapshot_model)):
            torch.save(snapshot_model[i].state_dict(),
                       gen_path(config.path, str(period), 'm' + str(i), filename=config.network + '.pkl'))

        torch.cuda.empty_cache()  # only to be monitored by nvidia-smi

        for i in range(len(snapshot_model)):
            snapshot_model[i].to(config.device)
            snapshot_model[i].eval()
        predict_y_val, real_y_val, valid_index_val = make_prediction(dataloader_val, sc_y, snapshot_model, config)
        predict_y_test, real_y_test, valid_index_test = make_prediction(dataloader_test, sc_y, snapshot_model, config)
        sharpe_head_val, alpha_head_val, dic_re = cal_performance(predict_y_val, real_y_val, val_date, valid_index_val,
                                                                  'val', writer, -1, config)
        sharpe_head_test, alpha_head_test, dic_re = cal_performance(predict_y_test, real_y_test, test_date,
                                                                    valid_index_test, 'test', writer, -1, config)
        model_renewal.append((-1, sharpe_head_val, sharpe_head_test, alpha_head_test))

    if config.eval_and_earlystop:
        with open(gen_path(config.path, filename='result.txt'), 'a') as result:
            last = model_renewal[-1]
            result.write(
                'Period %d, final round %d and its val sharpe %.2f & test sharpe %.2f & test alpha %.4f \n'
                % (period, last[0], last[1], last[2], last[3]))
    logfile.close()
    writer.close()


def eval_model(config, period, test_data):
    """
    After a model is trained, use it to give prediction csv on test set, so that can be used in backtest system
    """
    if config.network == 'MLPwithGAN':
        model = MLPwithGAN(config)
    elif config.network == 'MLP':
        model = MLP(config)
    elif config.network == 'LSTM':
        model = VanillaLSTM(config)
    elif config.network == 'CNN':
        model = CNNfeature(config)
    else:
        raise Exception('Unknown model type:{}'.format(config.network))

    if config.ensemble:
        m = model
        model = []

        for i in glob(gen_path(config.path, str(period)) + '/m*'):
            m.load_state_dict(
            torch.load(gen_path(i, filename=config.network + '.pkl')))
            m.to(config.device)
            m.eval()
            model.append(m)
    else:
        model.load_state_dict(
            torch.load(gen_path(config.path, str(period), 'model', filename=config.network + '.pkl')))
        model.to(config.device)
        model.eval()
    dataloader_test = test_data[0]
    test_date = test_data[1]
    test_symbol = test_data[2]
    sc_y = joblib.load(gen_path(config.path, str(period), 'scaler', filename='training_sc_y.pkl'))
    predict_y_test, real_y_test, valid_index_test = make_prediction(dataloader_test, sc_y, model, config)

    stock_score = pd.DataFrame()
    stock_score["symbol"] = test_symbol[valid_index_test]
    stock_score["score"] = predict_y_test
    stock_score['truth'] = real_y_test
    stock_score["date"] = test_date[valid_index_test]
    stock_score = stock_score.sort_values(by=["date"])
    stock_score.to_csv(gen_path(config.path, 'stock_score', filename=str(period) + '.csv'), index=False)


def cal_shap_process(period, config):
    """
    The whole process of using a trained model to calculate and store shapley values
    """
    print("Shapley calculation begins in periods: {} >>> pid={}, ppid={}".format(period, os.getpid(), os.getppid()))
    save_path = gen_path(config.shapley_path, filename=str(period) + '.npy')
    print('saving to', save_path)
    data_dict = generate_dataloader(config, period, val_length=config.val_length, train_length=config.train_length)
    if config.network == 'MLPwithGAN':
        model = MLPwithGAN(config)
    elif config.network == 'MLP':
        model = MLP(config)
    else:
        raise Exception('Unknown model type:{}'.format(config.network))
    model.load_state_dict(
        torch.load(gen_path(config.all_factor_path, str(period), 'model', filename=config.network + '.pkl')))
    model.to(config.device)
    model.eval()

    dataloader_train = data_dict['train'][0]
    train_iterator = iter(dataloader_train)
    dataloader_val = data_dict['val'][0]
    val_iterator = iter(dataloader_val)
    (_train_X, _, _) = next(train_iterator)
    background = torch.Tensor(_train_X).to(config.device)
    (_val_X, _, _) = next(val_iterator)
    sample = torch.Tensor(_val_X).to(config.device)
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(sample)
    features_shap = shap_values.mean(axis=0)
    sort_index = np.argsort(-np.abs(features_shap)).astype(int)

    save_txt = gen_path(config.shapley_path, filename=str(period) + '.txt')
    print('saving to ', save_txt)
    np.savetxt(save_txt, sort_index, fmt='%d')
    np.savetxt(save_txt.replace('.txt', '.npy'), sort_index, fmt='%d')

    del data_dict
    del model
    torch.cuda.empty_cache()
    print('--------Period {:d} shapley values calculation finished--------'.format(period))
    print("Shapley calculation process ends in periods: {}>>>".format(period))


def train_process(period, config):
    """
    The whole process
    :param period:
    :param config:
    :return:
    """
    print("Train process begins in period: {} >>> pid={}, ppid={}".format(period, os.getpid(), os.getppid()))
    data_dict = generate_dataloader(config, period, val_length=config.val_length, train_length=config.train_length)
    train_model(config, period, data_dict['train'], data_dict['val'], data_dict['test'])
    eval_model(config, period, data_dict['test'])  # todo: ban this now
    print("Time period %d finished, begin del" % period)
    del data_dict
    torch.cuda.empty_cache()
    print("train process ends in periods: {}>>>".format(period))


def main():
    config = Config()
    with open(gen_path(config.result_path, filename='foo.txt'), 'w') as f:
        f.write(' '.join(["%s = %s\n" % (k, v) for k, v in config.__dict__.items()]))
    print('Using gpu:', os.environ['CUDA_VISIBLE_DEVICES'])

    if config.train_full_model:
        print('start training full model')
        # Step.1 train all factor model
        config.all_factor_mode()
        with mp.Pool(config.num_train_process) as pool:
            pool.map(partial(train_process, config=config), config.train_periods)
        process_stock_score(config.path, config.train_periods)
        process_resulttxt(config.path)

    if config.cal_shapley:  # package shap does not support lstm
        print('start calculating shapley')
        config.cal_shap_mode()
        with mp.Pool(config.num_shap_process) as pool:
            pool.map(partial(cal_shap_process, config=config), config.train_periods)

    if config.shapley_filter_model:  # package shap does not support lstm
        print('start training filtered model')
        config.filter_factor_mode()
        with mp.Pool(config.num_train_process) as pool:
            pool.map(partial(train_process, config=config), config.train_periods)
        process_stock_score(config.path, config.train_periods)
        process_resulttxt(config.path)

    end_time = datetime.strftime(datetime.today(), '%Y%m%d.%H%M')
    print("Main process {:d} begun at {} ends at {}".format(os.getpid(), config.date_name, end_time))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()


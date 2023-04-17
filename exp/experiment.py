from data.data_loader import *
from exp.exp_basic import ExpBasic
from models.model import Informer, InformerStack, Probformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import *

import numpy as np
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings

warnings.filterwarnings('ignore')


class Experiment(ExpBasic):
    def __init__(self, args):
        super(Experiment, self).__init__(args)  # inherited the initialization of ExpBasic
        self.prob = False
        self.baseline = False
        if self.args.model == 'probformer' or self.args.model == 'transformer-p' or self.args.model == 'lstm-p':
            self.prob = True
            self.prob = True
        if self.args.model == 'transformer' or self.args.model == 'lstm' or \
                self.args.model == 'transformer-p' or self.args.model == 'lstm-p':
            self.baseline = True

    def _build_model(self):  # this function is called in exp_basic.__init__()
        model_dict = {
            'informer': Informer,  # ***
            'informerstack': InformerStack,
            'probformer': Probformer,
        }
        e_layers = self.args.s_layers if self.args.model == 'informerstack' else self.args.e_layers  # ***
        model = model_dict[self.args.model](
            self.args.enc_in,
            self.args.dec_in,
            self.args.c_out,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.factor,
            self.args.d_model,
            self.args.n_heads,
            e_layers,  # self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.dropout,
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.mix,
            self.device
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag, data=None):
        args = self.args

        data_dict = {
            'Boulder_charging_load': Dataset_Boulder,  # ***
            'EVnetNL_charging_load': Dataset_EVnetNL,  # ***
            'PALO_charging_load': Dataset_PALO,  # ***
            'Perth1_charging_load': Dataset_Perth1,  # ***
            'Perth2_charging_load': Dataset_Perth2,  # ***
        }
        if flag == 'meta_train':
            Data = data_dict[data]
        else:
            Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(  # Data == Dataset_ETT_hour/... (it's a Class)
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target='load',
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
        )
        print(data if data is not None else self.args.data, flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def baseline_train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):  # X_token and label are included in batch_y
                iter_count += 1

                model_optim.zero_grad()
                if self.prob:
                    [mu, sigma], true = self._baseline_process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = prob_loss(mu, sigma, true)
                else:
                    pred, true = self._baseline_process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = criterion(pred, true)  # loss of support set
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion) \
                if self.args.model == 'transformer' or self.args.model == 'lstm' else \
                self.prob_vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader, criterion) \
                if self.args.model == 'transformer' or self.args.model == 'lstm' else \
                self.prob_vali(test_data, test_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def reptile_train(self, setting):
        # all datasets
        train_data_all = []
        data_list = ['EVnetNL_charging_load', 'Boulder_charging_load', 'Perth1_charging_load']
        # data_list = ['PALO_charging_load', 'Boulder_charging_load', 'Perth1_charging_load', 'Perth2_charging_load']  # EVnetNL
        # data_list = ['EVnetNL_charging_load', 'PALO_charging_load', 'Perth1_charging_load', 'Perth2_charging_load']  # Boulder
        # data_list = ['EVnetNL_charging_load', 'PALO_charging_load', 'Boulder_charging_load']  # Perth2
        for d in data_list:
            train_data_all.append(
                list(self._get_data(flag='meta_train', data=d)))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()  # meta_lr = args.learning_rate
        criterion = self._select_criterion()  # MSELoss

        for epoch in range(self.args.meta_train_epochs):
            print('meta_epoch: {}'.format(epoch + 1))
            # randomly choose n tasks
            np.random.shuffle(train_data_all)
            batch_tasks = train_data_all[:self.args.task_batch_size]
            iter_count = 0
            train_loss = []
            weights_after_batch = []
            step_size = self.args.meta_lr * (1 - epoch / self.args.train_epochs)  # linear schedule

            if self.args.batch_mode:
                init_weights = copy.deepcopy(
                    self.model.state_dict())  # must use deepcopy, otherwise init_weights will be the same as weights_after
            self.model.train()

            for [train_data, train_loader] in batch_tasks:  # for each task Ti do:
                print('    task: {}'.format(train_data))
                train_steps = len(train_loader)

                if not self.args.batch_mode:
                    init_weights = copy.deepcopy(
                        self.model.state_dict()) 
                else:
                    self.model.load_state_dict(init_weights)
                self.model.train()

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    if self.args.model == 'probformer':
                        [mu, sigma], true = self._process_one_batch(
                            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                        support_loss = prob_loss(mu, sigma, true)
                    elif self.args.model == 'transformer-p':
                        [mu, sigma], true = self._baseline_process_one_batch(
                            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                        support_loss = prob_loss(mu, sigma, true)
                    elif self.args.model == 'transformer':
                        pred, true = self._baseline_process_one_batch(
                            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                        support_loss = criterion(pred, true)  # loss of support set
                    else:  # 'informer'
                        pred, true = self._process_one_batch(
                            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                        support_loss = criterion(pred, true)  # loss of support set
                    support_loss.backward()
                    model_optim.step()
                    train_loss.append(support_loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, support_loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                # update init_weights after each single sampled task, and then θ'j <-- θ
                if not self.args.batch_mode:
                    weights_after = self.model.state_dict()

                    self.model.load_state_dict(
                        {name: init_weights[name] + (weights_after[name] - init_weights[name]) * step_size
                         for name in init_weights})
                else:
                    weights_after_batch.append(copy.deepcopy(self.model.state_dict()))
            # update init param after all sampled tasks
            if self.args.batch_mode:
                self.model.load_state_dict(
                    {name: init_weights[name] + sum(weights_after[name] - init_weights[name]
                                                    for weights_after in
                                                    weights_after_batch) / self.args.task_batch_size * step_size
                     for name in init_weights})

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                if self.prob:  # probformer
                    [mu, sigma], true = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = prob_loss(mu, sigma, true)
                else:  # informer
                    pred, true = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion) if not self.prob \
                else self.prob_vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader, criterion) if not self.prob \
                else self.prob_vali(test_data, test_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark) \
                if not self.baseline else \
                self._baseline_process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def prob_vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            [mu, sigma], true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark) \
                if self.args.model == 'probformer' else \
                self._baseline_process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = prob_loss(mu.detach().cpu(), sigma.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def meta_test(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()  # MSELoss

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # ========================================fine-tune in meta test========================================= #
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                if self.args.model == 'probformer':
                    [mu, sigma], true = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = prob_loss(mu, sigma, true)
                elif self.args.model == 'transformer-p':
                    [mu, sigma], true = self._baseline_process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = prob_loss(mu, sigma, true)
                elif self.args.model == 'transformer':
                    pred, true = self._baseline_process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = criterion(pred, true)
                else:  # 'informer'
                    pred, true = self._process_one_batch(
                        train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                    loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if self.prob:
                vali_loss = self.prob_vali(vali_data, vali_loader)
            else:
                vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # ========================================final test in meta test========================================= #
        if self.prob:
            crps = self.test(setting)
            return crps
        else:
            self.test(setting)

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()

        mus = []
        sigmas = []
        preds = []
        trues = []

        if self.prob:
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                [mu, sigma], true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark) \
                    if self.args.model == 'probformer' else \
                    self._baseline_process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                mus.append(mu.detach().cpu().numpy())
                sigmas.append(sigma.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

            mus = np.array(mus)
            sigmas = np.array(sigmas)
            trues = np.array(trues)
            print('test shape:', mus.shape, trues.shape)
            mus = mus.reshape(-1, mus.shape[-2], mus.shape[-1])
            sigmas = sigmas.reshape(-1, sigmas.shape[-2], sigmas.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', mus.shape, trues.shape)

            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            crps, rho50, rho90, mae, rmse = probEvaluator(mus, sigmas, trues)
            print(
                'RMSE, rho50, rho90, crps are: {:.3f} & {:.3f} & {:.3f} & {:.3f}'.format(rmse, rho50, rho90, crps))

            np.save(folder_path + 'metrics.npy', np.array([rmse, rho50, rho90, crps]))
            np.save(folder_path + 'preds.npy', np.array([mus, sigmas]))
            np.save(folder_path + 'true.npy', trues)
            return crps

        else:
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark) \
                    if self.args.model == 'informer' else \
                    self._baseline_process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('MAE, MSE, RMSE, MAPE, MSPE are: {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}'.format(
                mae, mse, rmse, mape, mspe))

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'preds.npy', preds)
            np.save(folder_path + 'true.npy', trues)

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def _baseline_process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        if self.args.model == 'transformer' or self.args.model == 'transformer-p':
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()

            # decoder input
            if self.args.padding == 0:
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            elif self.args.padding == 1:
                dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            # The last label_len sequence in the input seq & padding sequence
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        else:  # LSTM or LSTM-P
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            batch_x = batch_x.reshape([batch_x.size(0), -1, batch_x.size(1)]).to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, dec_inp)[0]
                else:
                    outputs = self.model(batch_x, dec_inp)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, dec_inp)[0]
            else:
                if self.args.model == 'transformer' or self.args.model == 'transformer-p':
                    outputs = self.model(batch_x, dec_inp)
                else:
                    outputs = self.model(batch_x)

        if self.args.inverse:  # it's not post-processing, but can be seen as a part of the network
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

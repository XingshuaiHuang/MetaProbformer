import argparse
from utils.tools import seed_everything
from exp.experiment import Experiment
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # ***

"""
The code is based on the codebase of Informer (https://ojs.aaai.org/index.php/AAAI/article/view/17325).
"""

# ***********************************************************************
# Hyperparameters
# ***********************************************************************

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='probformer',
                    help='model of experiment')
parser.add_argument('--meta', type=str, required=True, default='reptile',
                    help='meta-learning method')

# ---------------------- data settings ---------------------- #
parser.add_argument('--data', type=str, required=True, default='PALO_charging_load', help='data')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# ---------------------- model settings ---------------------- #
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='file list')

# ---------------------- training settings ---------------------- #
parser.add_argument('--seed', type=int, default=1995, help='random seed')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
parser.add_argument('--meta_train_epochs', type=int, default=10, help='train epochs for meta-learning')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--task_batch_size', type=int, default=2, help='batch size of sampled tasks')
parser.add_argument('--patience', type=int, default=8, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--meta_lr', type=float, default=0.1, help='meta learning rate')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--des', type=str, default='exp',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_false', help='inverse output data', default=True)
parser.add_argument('--plot_results', action='store_true', help='plot final predictions', default=False)
parser.add_argument('--batch_mode', action='store_true', help='plot final predictions', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--random_search', action='store_true', help='hyperparameter optimization', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# Set augments by using data name
data_parser = {
    'PALO_charging_load':{'data':'PALO_charging_load.csv','T':'load','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'Boulder_charging_load':{'data':'Boulder_charging_load.csv','T':'load','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'Perth2_charging_load':{'data':'Perth2_charging_load.csv','T':'load','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'Perth1_charging_load':{'data':'Perth1_charging_load.csv','T':'load','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'EVnetNL_charging_load':{'data':'EVnetNL_charging_load.csv','T':'load','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]  # 1,1,1 for S:univariate predict univariate

args.detail_freq = args.freq
args.freq = args.freq[-1:]
start_time = time.time()

print('\n================================')
print('Args in experiment:')
print(args)


# ***********************************************************************
# Experiment
# ***********************************************************************
seed_everything(seed=args.seed)

Exp = Experiment

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_mlr{}_mte{}_te{}_bm{}_dp{}_sd{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}' \
        .format(args.model, args.meta, args.data, args.meta_lr, args.meta_train_epochs,
                args.train_epochs, args.batch_mode, args.dropout, args.seed,
                args.features, args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers,
                args.d_ff, args.attn, args.factor, args.embed,
                args.distil, args.mix, args.des, ii)

    # set experiments
    exp = Exp(args)

    # train
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    if args.meta == 'reptile':
        exp.reptile_train(setting)
    elif args.model == 'lstm' or args.model == 'transformer' or args.model == 'lstm-p' or args.model == 'transformer-p':
        exp.baseline_train(setting)
    else:
        exp.train(setting)

    # test
    print('>>>>>>>testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    if args.meta == 'reptile':
        exp.meta_test(setting)
    else:
        exp.test(setting)

print('*************** Total time: {} ***************'.format(time.time() - start_time))
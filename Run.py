import os
import sys
__file__ = '../data/PEMS04.npz'
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

sys.argv=['']
del sys
#*************************************init_embedding************************************#
import os
import numpy as np
from lib.dataloader import get_train_adj_matrix_and_embeddings

# Configuration
dataset_name = "PEMSD4"
val_ratio = 0.2
test_ratio = 0.2
prediction_length = 1  # Set prediction length
lambda1 = 0.02
embedding_dim = 10  # Set embedding dimensions

# Flag to determine which initialization to use
use_precomputed = True  # Set to False to use random initialization

# File path for the precomputed embeddings
E_init_file_name = f"E_init/E_init_{dataset_name}_pre_len{prediction_length}.npy"

if use_precomputed:
    if os.path.exists(E_init_file_name):
        # If the file exists, load the precomputed embeddings
        precomputed_embeddings = np.load(E_init_file_name)
        print(f"Loaded precomputed embeddings from {E_init_file_name}")
    else:
        # Otherwise, generate the adjacency matrix and embeddings, then save them
        _, precomputed_embeddings = get_train_adj_matrix_and_embeddings(
            dataset_name, val_ratio, test_ratio, prediction_length, lambda1, embedding_dim)
        np.save(E_init_file_name, precomputed_embeddings)
        print(f"Saved precomputed embeddings to {E_init_file_name}")
else:
    precomputed_embeddings = None
    print("Using random initialization for embeddings.")

#*************************************************************************#
import torch
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.SGCRN import SGCRN_RD as Network
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch
#*************************************************************************#
#get configuration
Mode = 'Train'
DEBUG = 'True'
DATASET = 'PEMSD4'      #PEMSD4 or PEMSD8
DEVICE = 'cuda:0'
MODEL = 'SGCRN'
config_file = 'model/PEMSD4_SGCRN.conf'  # Assuming the file is in the same directory as your script
config = configparser.ConfigParser()
config.read(config_file)


from lib.metrics import MAE_torch
def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss



#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--tod', default=config['data']['tod'], type=eval)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
#args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args = args.parse_args()
init_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

#init model
# Initialize the model with precomputed embeddings
model = Network(args, precomputed_embeddings=precomputed_embeddings)
model = model.to(args.device)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
print_model_parameters(model, only_num=False)

#load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=False,
                                                               weather=False, single=False)
##########################Trainer##################################

if args.loss_func == 'mask_mae':
    loss = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss().to(args.device)
else:
    raise ValueError

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)

#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
args.log_dir = log_dir

# # GPU
# if torch.cuda.is_available():
#     args.device = 'cuda:0'
# else:
#     print("CUDA is not available. Switching to CPU.")
#     args.device = 'cpu'

# model = model.to(args.device)

# #start training
# trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
#                   args, lr_scheduler=lr_scheduler)

# if args.mode.lower() == 'train':
#     trainer.train()
# elif args.mode.lower() == 'test':
#     # Load the model on CPU
#     model.load_state_dict(torch.load('../pre-trained/.pth'.format(args.dataset), map_location=torch.device('cpu')))
#     model.to(torch.device('cpu'))
#     print("Load saved model")
#     trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
# else:
#     raise ValueError("Invalid mode specified: {}".format(args.mode))
    
#CPU
#start training
trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)

if args.mode.lower() == 'train':
    trainer.train()
elif args.mode.lower() == 'test':
    # Load the model on CPU
    model.load_state_dict(torch.load('../pre-trained/.pth'.format(args.dataset), map_location=torch.device('cpu')))
    model.to(torch.device('cpu'))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError("Invalid mode specified: {}".format(args.mode))

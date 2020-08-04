import torch
import torch.nn as nn
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from utils.config import cfg

from models import WENet
from step import train_we
from dataloaders.datasets import WE_Data, pad_collate_we
import pdb

random.seed(cfg.manualSeed)
np.random.seed(cfg.manualSeed)
torch.manual_seed(cfg.manualSeed)
torch.cuda.manual_seed(cfg.manualSeed)
#torch.set_num_threads(4)

if cfg.CUDA:    
    torch.cuda.manual_seed(cfg.manualSeed)  
    torch.cuda.manual_seed_all(cfg.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):   
    np.random.seed(cfg.manualSeed + worker_id)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training the model for word boundary discovery')

    parser.add_argument('--data_path', type=str, default= 'X:/staff-bulk/ewi/insy/MMC/xinsheng/data/coco/',
                        help='directory of database')   
    parser.add_argument('--save_root',type=str,default='outputs/P10',
                        help='path for saving model and results')    
    # parameters
    parser.add_argument("--epoch", type=int, default=120,
                        help="max epoch")
    parser.add_argument("--evaluation",type=bool, default=True,
                        help='True for evaluation only')
    parser.add_argument("--start_epoch",type=int,default=90,
                        help='resume the pre-trained parameter')
    parser.add_argument("--optim", type=str, default="adam",
                        help="training optimizer", choices=["sgd", "adam"])
    parser.add_argument('--batch_size', default=4, type=int, 
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr_decay', default=100, type=int, metavar='LRDECAY',
                        help='Divide the learning rate by 10 every lr_decay epochs')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')  
    parser.add_argument('--LAMDA',default=5,type=float,
                        help='smooth3')
    parser.add_argument('--penalty',default=10,type=float,
                        help='penalty for mel spec reconstruction loss')


    # evaluation
    parser.add_argument('--threshold',default=0.055,type=float,
                        help='threshold value for select predicted value')
    parser.add_argument('--BK_train',default=0,type=int,
                        help='neighbor number of the ground-truth boundary that considered as the boundary during training ')
    parser.add_argument('--BK',default=2,type=int,
                        help='predicted boundary is considered to be correct is it is whin BK frames of the ground-truth boundary')

    args = parser.parse_args()
    
    if args.LAMDA != '':
        cfg.WD.smooth3 = args.LAMDA
    
    dataset = WE_Data(args.data_path,args,'train')
    dataset_val = WE_Data(args.data_path,args,'val')
   
    
    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            drop_last=True, shuffle=True,num_workers=cfg.workers,collate_fn=pad_collate_we,worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=256,
        drop_last=False, shuffle=False,num_workers=cfg.workers,collate_fn=pad_collate_we,worker_init_fn=worker_init_fn) 
    
    Encoder = WENet.Encoder(cfg.WD.input_size,cfg.WD.hidden_size,args)
    Decoder = WENet.Decoder_BLSTM(args,cfg.WD.hidden_size,40)

    
    train_we.train(Encoder,Decoder,train_loader,args)
    


    
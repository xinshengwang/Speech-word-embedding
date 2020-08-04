import torch
import torch.nn as nn
import os
import numpy as np
from utils.config import cfg
from utils.util import adjust_learning_rate, AverageMeter, word_prediction, reconstruction_loss
import pdb
import matplotlib.pyplot as plt
import librosa.display as display


def train(encoder,decoder,train_loader,args):
    if cfg.CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    exp_dir = args.save_root    
    save_model_path = os.path.join(exp_dir,'models')
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    epoch = args.start_epoch
    if epoch != 0:
        decoder.load_state_dict(torch.load("%s/models/WE_decoder%d.pth" % (exp_dir,epoch)))
        encoder.load_state_dict(torch.load("%s/models/WE_encoder%d.pth" % (exp_dir,epoch)))
        print('loaded parametres from epoch %d' % epoch)

    trainables_en = [p for p in encoder.parameters() if p.requires_grad]
    trainables_de = [p for p in decoder.parameters() if p.requires_grad]
    trainables = trainables_en + trainables_de

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(trainables, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    
    criterion_mse = nn.MSELoss() 
    criterion_bce = nn.BCELoss()
    
    save_file = os.path.join(exp_dir,'results_WD.txt')
    while epoch <= args.epoch:
        loss_meter = AverageMeter()
        epoch += 1
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)

        encoder.train()
        decoder.train()

        for i, (audio,mask,length) in enumerate(train_loader):
            loss = 0
            word_len = mask.sum(1)
            word_len = word_len.float().cuda()
            audio = audio.float().cuda()
            mask = mask.float().cuda()
            optimizer.zero_grad()

            audio_features,word_nums = encoder(audio,mask,length)
            recons_audio = decoder(audio_features,mask,length)
            loss = reconstruction_loss(audio,recons_audio,length,args)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(),args.batch_size)

            if i%5 == 0:
                print('iteration = %d | loss = %f '%(i,loss))
                
        
        if epoch % 5 == 0:
            torch.save(encoder.state_dict(),
                "%s/models/WE_encoder%d.pth" % (exp_dir,epoch))
            torch.save(decoder.state_dict(),
                "%s/models/WE_decoder%d.pth" % (exp_dir,epoch))
            # true_num,total_predict,real_num,P = evaluation(encoder,decoder,val_loader,val_image_loader,args)
            # info = "epoch {} | loss {:.2f} | True {} | Predict {} | Real {} | P {:.2%}\n".format(epoch,loss,true_num,total_predict,real_num,P)
            info = "epoch {} | loss {:.2f} \n".format(epoch,loss_meter.avg)
            print(info)
            with open(save_file,'a') as f:
                f.write(info)
        

def evaluation(encoder,decoder,val_loader,val_image_loader,value):
    encoder.eval()
    decoder.eval()
    image_features = []
    image_labels =[]
    cat_features = []
    cat_labels = []
    
    # print('start loading image')
    for i, (image, label) in enumerate(val_image_loader):
        image = image.float().cuda()
        image_feature = decoder(image)
        image_feature = image_feature.data.to('cpu')
        image_feature = image_feature.mean((2,3))
        image_features.append(image_feature)
        image_labels.append(label)
        # print(i)
    # print('finish loading image')
    image_features = torch.cat(image_features).to('cpu')
    labels = torch.cat(image_labels)
    categoris = labels.unique()
    for cat in categoris:
        index = np.where(labels==cat)
        cat_feature = (image_features[index]).mean(0)
        cat_feature = cat_feature.unsqueeze(0)
        cat_features.append(cat_feature)
        cat_labels.append(cat)
    cat_features = torch.cat(cat_features)
    cat_labels = np.array(cat_labels)
    cat_labels = torch.from_numpy(cat_labels)
    # print('start loading speech')
    for i, (image,audio,mask,length,image_id,labels) in enumerate(val_loader):
        word_len = mask.sum(1)
        word_len = word_len.float()
        audio = audio.float().cuda()
        mask = mask.float().cuda()
        audio_features,word_nums = encoder(audio,mask,length)
        audio_features = audio_features.to('cpu')
        audio_features = audio_features.transpose(2,1)
        # pdb.set_trace()
        true_num,total_predict,real_num = word_prediction(audio_features,labels,cat_features,cat_labels,value)
        # print(i)
        if i==0:
            true_nums = true_num
            total_predicts = total_predict
            real_nums = real_num
        else:
            true_nums += true_num
            total_predicts += total_predict
            real_nums += real_num
    pdb.set_trace()
    P =  true_nums.sum() / total_predicts.sum()

    return true_nums.sum(),total_predicts.sum(),real_nums.sum(),P

def get_predicted_boundary(predict,k):
    max_pool = nn.MaxPool1d(2*k+1,1,padding=2).cuda()
    ext_predict = predict.unsqueeze(1)
    max_predict = max_pool(ext_predict).squeeze()
    max_mask = ((predict - max_predict) >= 0).int().to('cpu')
    sub_mask = max_mask -1
    predict = predict.to('cpu').detach() 
    predict = predict * max_mask + sub_mask
    return predict

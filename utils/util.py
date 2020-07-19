import math
import pickle
import numpy as np
import torch
from utils.config import cfg
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb


def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every lr_decay epochs"""
    lr = base_lr * (0.5 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def words_loss(img_features, words_emb, class_ids, cap_len):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    cap_len = cap_len.int()
    batch_size = img_features.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size)))
    labels = labels.cuda()  

    masks = []
    att_maps = []
    similarities = []
    class_ids =  class_ids.data.cpu().numpy()
    for i in range(batch_size):
        if class_ids is not None:            
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_len[i]
        # -> 1 x nef x words_num
        word = words_emb[i,:,:words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, cfg.WD.smooth1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.WD.smooth2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)   
    if class_ids is not None:
        masks = np.concatenate(masks, 0)        
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        masks = masks.to(torch.bool)
        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.WD.smooth3
    
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0 + loss1, att_maps

def text_fuc(x):
    x = 0
    return x

def get_att_maps(img_features, words_emb,cap_len):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    cap_len = cap_len.int()
    batch_size = img_features.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size)))
    labels = labels.cuda()  

    att_maps = []
    for i in range(batch_size):
        # Get the i-th text description
        words_num = cap_len[i]
        # -> 1 x nef x words_num
        word = words_emb[i,:,:words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, cfg.WD.smooth1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        
    return att_maps




def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax(dim=1)(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    #  Eq. (9)
    attn = attn  #* gamma1
    attn = nn.Softmax(dim=1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


def word_prediction(audio_output,labels,cat_features,cat_labels,value):      
    # using consine similarity    
    # pdb.set_trace()
    Correct_num = torch.zeros(cat_labels.shape[0])
    Total_pred =  torch.zeros(cat_labels.shape[0])
    for i in range(audio_output.shape[0]):
        audio_feature = audio_output[i]
        img_f = normalizeFeature(cat_features)
        aud_f = normalizeFeature(audio_feature) 
        S = img_f.mm(aud_f.t())
        
        # for audio to image retrieval
        S_T = S.T * 10.0
        score = F.softmax(S_T,dim=-1)
        sorted_scores, indx_A2I = torch.sort(score,dim=1,descending=True)
        max_scores = sorted_scores[:,0]
        mask = (max_scores > value).int()
        indx = torch.where(mask==1)
        # pdb.set_trace()
        class_sorted_A2I = cat_labels[indx_A2I][:,0]

        pred_labels = torch.zeros(len(labels[i]))
     
        class_sorted_A2I = class_sorted_A2I[indx]
        pred_labels[class_sorted_A2I] = 1

        Correct_num += pred_labels * labels[i]
        Total_pred += pred_labels
    # pdb.set_trace()
    Real_num = labels.sum(dim=0)      # Ground truth number

    return Correct_num, Total_pred, Real_num

def normalizeFeature(x):	
    
    x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide\
    feature_norm = torch.sum(x**2, axis=1)**0.5 # l2-norm
    feat = x / feature_norm.unsqueeze(-1)
    return feat


def reconstruction_loss(input,output,length,args):
    input.requires_grad = False
    seq = torch.from_numpy(np.arange(input.shape[1]))
    seq = seq.unsqueeze(0).repeat(input.shape[0],1)
    length = length.unsqueeze(-1).repeat(1,input.shape[1])
    mask = (seq <= length).int().unsqueeze(-1).repeat(1,1,input.shape[-1]).cuda()
    input = input*mask
    output = output*mask
    # loss = ((input - output) ** 2).sum() / (mask.sum() + 1e-7)
    loss = nn.MSELoss()(input*args.penalty,output*args.penalty)
    return loss
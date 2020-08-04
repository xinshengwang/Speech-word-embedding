import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import cfg
import numpy as np

class Encoder(nn.Module):

    def __init__(self,input_size,hidden_size,args,num_layers=1,dropout=0.0,bidirectional=True):
        super(Encoder,self).__init__()
        self.rnn = nn.LSTMCell(input_size,hidden_size)
        self.hidden_size = hidden_size

    def init_hidden_state(self,batch_size):
        # mean = input.mean(dim=1)
        h = torch.zeros(batch_size,self.hidden_size).cuda()
        c = torch.zeros(batch_size,self.hidden_size).cuda()
        return h, c

    def forward(self,input,mask,length):
        """
        take the work boundary as the begining of the next word.
        """
        max_length = length.max()
        batch_size = input.shape[0]
        h,c  = self.init_hidden_state(batch_size)
        embeddings = torch.zeros(batch_size,max_length,self.hidden_size).cuda()
        end_indx = length.unsqueeze(-1).long().cuda() - 1
        mask[:,0] = 0
        mask = mask.scatter_(1,end_indx,1)
        for t in range(max_length):
            pad_mask = mask[:,t].unsqueeze(-1).repeat(1,self.hidden_size)
            h = h*pad_mask[:h.shape[0],:]
            c = c*pad_mask[:h.shape[0],:]
            batch_size_t = sum([l >t for l in length])
            h,c = self.rnn(input[:batch_size_t,t,:], (h[:batch_size_t],c[:batch_size_t])
                            )
            embeddings[:batch_size_t,t,:] = h
        
        word_nums = mask.sum(1)
        max_word_num = word_nums.max().int()
        i = 0

        # take the
        for num in word_nums:
            p = (num - word_nums.max()).int()
            # if mask[i,-1] != 0:
            #     p = p-1
            #     mask[i,p:] = 1
            if p != 0:
                mask[i,p:] = 1
                while mask[i].sum()<word_nums.max():
                    p -= 1
                    mask[i,p:] = 1
            i += 1
        index = mask.nonzero() # index[0,:] batch position, index[:,1] frame position
        index = index[:,0] * mask.shape[1] + index[:,1] - 1 # we take the frame of the boundary as the begaining of next word
        flat = embeddings.view(-1,embeddings.shape[-1])
        word_embeddings = flat[index.long()]
        output = word_embeddings.view(batch_size,max_word_num,-1)
        # output = output.transpose(2,1)

        return output, word_nums

class Decoder(nn.Module):
    def __init__(self,args,input_size,hidden_size=80,num_layers=1,dropout=0.0,bidirectional=True):
        super(Decoder,self).__init__()
        self.rnn = nn.LSTMCell(input_size,hidden_size)
        self.hidden_size = hidden_size

    def init_hidden_state(self,batch_size):
        # mean = input.mean(dim=1)
        h = torch.zeros(batch_size,self.hidden_size).cuda()
        c = torch.zeros(batch_size,self.hidden_size).cuda()
        return h, c

    def get_frame_feature(self,input,word_id):
        '''
        input: the word level embedding from the encoder
        word_id: frame level word id (the frame belongs to which word feature)
        output: frame level embedding--each frame is a word embedding
        '''
        # input = input.transpose(2,1)
        batch_size = input.shape[0]
        mask = torch.from_numpy(np.arange(batch_size)).unsqueeze(1).repeat(1,word_id.shape[1]).cuda()
        index = input.shape[1] * mask + word_id
        input_flat = input.view(-1,input.shape[-1])
        select = input_flat[index.long()]
        output = select.view(batch_size,mask.shape[1],-1)
        return output 
        


    def forward(self,input,mask,length):
        """
        take the work boundary as the begining of the next word.
        """
        max_length = length.max()
        batch_size = input.shape[0]
        h,c  = self.init_hidden_state(batch_size)
        filterbank = torch.zeros(batch_size,max_length,self.hidden_size).cuda()
        end_indx = length.unsqueeze(-1).long().cuda() - 1
        mask[:,0] = 0
        mask = mask.scatter_(1,end_indx,1)
        
        # get the word id of each frame
        # shape(batch_size, max_length) [0,0,..,1,1,1...,...]
        mask_sum = torch.zeros(batch_size,1).cuda()
        for i in range(max_length):
            mask_slice = mask[:,i].unsqueeze(-1)
            mask_sum += mask_slice
            if i == 0:
                word_id = mask_sum.clone()
            else:
                word_id = torch.cat([word_id,mask_sum],1)   
        
        word_id[:,-1] = word_id[:,-2]
        
        input = self.get_frame_feature(input,word_id)

        for t in range(max_length):               
            pad_mask = mask[:,t].unsqueeze(-1).repeat(1,self.hidden_size)
            h = h*pad_mask[:h.shape[0],:]
            c = c*pad_mask[:h.shape[0],:]
            batch_size_t = sum([l >t for l in length])

            h,c = self.rnn(input[:batch_size_t,t,:], (h[:batch_size_t],c[:batch_size_t])
                            )
            filterbank[:batch_size_t,t,:] = h

        return filterbank

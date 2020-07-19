import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import cfg

class RNN(nn.Module):

    def __init__(self,input_size,hidden_size,args,num_layers=1,dropout=0.0,bidirectional=True):
        super(RNN,self).__init__()
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
        output = output.transpose(2,1)



        return output, word_nums

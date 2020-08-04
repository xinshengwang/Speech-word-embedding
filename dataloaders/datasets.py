from utils.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate

import os
import sys
import json
import pickle
import numpy as np
import numpy.random as random

from PIL import Image
import torchvision.transforms as transforms

def pad_collate(batch):
    max_input_len = float('-inf')  
    
    for elem in batch:
        mel, target, length = elem
        max_input_len = max_input_len if max_input_len > length else length       

    for i, elem in enumerate(batch):
        mel, target, length = elem
        input_length = mel.shape[1]
        input_dim = mel.shape[0]

        pad_mel = np.zeros((input_dim,max_input_len), dtype=np.float)
        pad_mel[:input_dim, :input_length] = mel       

        pad_target = np.zeros(max_input_len,dtype=np.float)
        mask = np.zeros(max_input_len,dtype=np.float)
        pad_target[:input_length] = target
        mask[:input_length] = 1.0
        pad_mel = pad_mel.transpose(1,0)
        batch[i] = (pad_mel, pad_target, mask, length)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))

    batch.sort(key=lambda x: x[-1], reverse=True)

    return default_collate(batch)

def pad_collate_wd(batch):
    max_input_len = float('-inf')  
    
    for elem in batch:
        image, mel, target, length, image_id = elem
        max_input_len = max_input_len if max_input_len > length else length       

    for i, elem in enumerate(batch):
        image, mel, target, length, image_id = elem
        input_length = mel.shape[1]
        input_dim = mel.shape[0]

        pad_mel = np.zeros((input_dim,max_input_len), dtype=np.float)
        pad_mel[:input_dim, :input_length] = mel       

        pad_target = np.zeros(max_input_len,dtype=np.float)
        mask = np.zeros(max_input_len,dtype=np.float)
        pad_target[:input_length] = target
        mask[:input_length] = 1.0
        pad_mel = pad_mel.transpose(1,0)
        batch[i] = (image,pad_mel, pad_target, length, image_id)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))

    batch.sort(key=lambda x: x[-2], reverse=True)

    return default_collate(batch)

def pad_collate_wd_val(batch):
    max_input_len = float('-inf')  
    
    for elem in batch:
        image, mel, target, length, image_id, labels = elem
        max_input_len = max_input_len if max_input_len > length else length       

    for i, elem in enumerate(batch):
        image, mel, target, length, image_id, labels = elem
        input_length = mel.shape[1]
        input_dim = mel.shape[0]

        pad_mel = np.zeros((input_dim,max_input_len), dtype=np.float)
        pad_mel[:input_dim, :input_length] = mel       

        pad_target = np.zeros(max_input_len,dtype=np.float)
        mask = np.zeros(max_input_len,dtype=np.float)
        pad_target[:input_length] = target
        mask[:input_length] = 1.0
        pad_mel = pad_mel.transpose(1,0)
        batch[i] = (image,pad_mel, pad_target, length, image_id,labels)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))

    batch.sort(key=lambda x: x[-3], reverse=True)

    return default_collate(batch)


def pad_collate_we(batch):
    max_input_len = float('-inf')  
    
    for elem in batch:
        mel, target, length = elem
        max_input_len = max_input_len if max_input_len > length else length       

    for i, elem in enumerate(batch):
        mel, target, length = elem
        input_length = mel.shape[1]
        input_dim = mel.shape[0]

        pad_mel = np.zeros((input_dim,max_input_len), dtype=np.float)
        pad_mel[:input_dim, :input_length] = mel       

        pad_target = np.zeros(max_input_len,dtype=np.float)
        mask = np.zeros(max_input_len,dtype=np.float)
        pad_target[:input_length] = target
        mask[:input_length] = 1.0
        pad_mel = pad_mel.transpose(1,0)
        batch[i] = (pad_mel, pad_target, length)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))

    batch.sort(key=lambda x: x[-1], reverse=True)

    return default_collate(batch)



def get_imgs(img_path, imsize=256, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
   
    if transform is not None:
        img = transform(img)
    return normalize(img)



class WBD_Data(data.DataLoader):
    def __init__(self, data_path,args,split='train'):
        self.args = args
        self.split = split
        if split=='train':
            self.data_dir = os.path.join(data_path,'train2014')
        else:
            self.data_dir = os.path.join(data_path,'val2014')
        self.filenames = self.load_filelnames(self.data_dir)

    def load_filelnames(self,data_dir):
        if self.split == 'train':
            path = data_dir + '/filenames/' + 'Bruce_1.0_None.json'
        else:
            path = data_dir + '/filenames/' + 'Bruce_1.0_None_uniqueImgID.json'
        with open(path,'rb') as f:
            data = json.load(f)
        return data

    def load_json(self,path):
        with open(path,'rb') as f:
            data = json.load(f)
        return data

    def load_target(self,wav_duration,mel_len,timecode):
        target = np.zeros(mel_len)
        times = []
        for item in timecode:
            if 'WORD' in item:
                time = item[0]
                times.append(time)
        times = np.array(times)

        positions = times * mel_len / (wav_duration*1000)
        positions = np.around(positions).astype(np.int32)
        if self.split == 'train':
            if self.args.BK_train == 0:
                target[positions] = 1
            else:
                pad_positions = list(positions)
                
                for i in range(self.args.BK_train):
                    k = i+1
                    positions_right = positions[:-1] + k
                    positions_left = positions[1:] - k
                    pad_positions = pad_positions + list(positions_left) + list(positions_right)
                target[pad_positions] = 1
        else:
            pad_positions = list(positions)
            for i in range(self.args.BK):
                k = i+1
                positions_right = positions[:-1] + k
                positions_left = positions[1:] - k
                pad_positions = pad_positions + list(positions_left) + list(positions_right)
            target[pad_positions] = 1
        return target

    def __getitem__(self,index):
        data_dict = self.filenames[index]
        wav_name = data_dict['wavFilename']
        mel_name = wav_name.replace('.wav','.npy')
        mel_path = self.data_dir + '/mel/' + mel_name 
        mel =  np.load(mel_path,allow_pickle=True)
        json_path = self.data_dir + '/json/' + wav_name.replace('.wav','.json')
        json_dict = self.load_json(json_path)

        wav_duration = json_dict['duration']
        mel_len = mel.shape[1]
        timecode = json_dict['timecode']

        target = self.load_target(wav_duration,mel_len,timecode)
        
        return mel, target, mel_len


    def __len__(self):
        return len(self.filenames)

class WD_Data(data.DataLoader):
    """
    imgs: image
    mel: image speech caption mel feature  
    mask: target, for training: a sequence of 0 or 1, 1 if the frame is boundary or 0;
    """
    def __init__(self, data_path,args,split='train',
                 img_size=256,
                 transform=None,
                 ):
        self.args = args
        self.split = split
        aduio_path = os.path.join(data_path,'audio')
        image_path = os.path.join(data_path,'Image')
        if split=='train':
            self.audio_dir = os.path.join(aduio_path,'train2014')
            self.image_dir = os.path.join(image_path,'train2014')
        else:
            self.audio_dir = os.path.join(aduio_path,'val2014')
            self.image_dir = os.path.join(image_path,'val2014')
            self.single_img_dir = image_path + '/Single/' + 'val2014'
            
        self.filenames = self.load_filelnames(self.audio_dir)[:1000]

        self.imsize = img_size
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def load_filelnames(self,data_dir):
        if self.split == 'train':
            path = data_dir + '/filenames/' + 'Bruce_1.0_None.json'
        else:
            path = data_dir + '/filenames/' + 'Bruce_1.0_None_uniqueImgID_with_categories.json'
        with open(path,'rb') as f:
            data = json.load(f)
        return data

    def load_json(self,path):
        with open(path,'rb') as f:
            data = json.load(f)
        return data

    def load_target(self,wav_duration,mel_len,timecode):
        target = np.zeros(mel_len)
        times = []
        for item in timecode:
            if 'WORD' in item:
                time = item[0]
                times.append(time)
        times = np.array(times)

        positions = times * mel_len / (wav_duration*1000)
        positions = np.around(positions).astype(np.int32)
        target[positions] = 1
        
        return target
    def get_image_path(self,image_id):
        ID = str(image_id)
        name = ID.zfill(12)
        if self.split == 'train':
            image_name = 'COCO_train2014_' + name + '.jpg'
        else:
            image_name = 'COCO_val2014_' +  name + '.jpg'
        img_path = os.path.join(self.image_dir,image_name)
        return img_path

    def transfer_id_to_continuous(self,old_id):
        path = os.path.join(self.single_img_dir,'skip2conti.json')
        with open(path,'r') as f:
            dic = json.load(f)
        new_id = dic[old_id]
        return new_id

    def transfer_id_to_oneHot(self,cat_ids):
        new_ids = [self.transfer_id_to_continuous(str(cat)) for cat in cat_ids]
        label = np.zeros(80)
        label[new_ids] = 1
        return label
    def __getitem__(self,index):
        data_dict = self.filenames[index]
        wav_name = data_dict['wavFilename']
        mel_name = wav_name.replace('.wav','.npy')
        mel_path = self.audio_dir + '/mel/' + mel_name 
        mel =  np.load(mel_path,allow_pickle=True)
        
        image_id = data_dict['imageID']
        image_path = self.get_image_path(image_id)

        imgs = get_imgs(image_path,self.imsize,self.transform,self.norm) 

        json_path = self.audio_dir + '/json/' + wav_name.replace('.wav','.json')
        json_dict = self.load_json(json_path)

        wav_duration = json_dict['duration']
        mel_len = mel.shape[1]
        timecode = json_dict['timecode']
        mask = self.load_target(wav_duration,mel_len,timecode)

        if self.split == 'train':
            return imgs, mel, mask, mel_len, image_id
        else:
            cat_ids = data_dict['categories']
            labels = self.transfer_id_to_oneHot(cat_ids)
            return imgs, mel, mask, mel_len, image_id, labels

    def __len__(self):
        return len(self.filenames)

class WD_Data_img(data.DataLoader):
    def __init__(self, data_path,args,transform=None):
        self.args = args
        image_path = os.path.join(data_path,'Image')
        self.image_dir = image_path +'/Single/val2014'        
        self.filenames = self.load_filelnames(self.image_dir)
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def load_filelnames(self,data_dir):        
        path = os.path.join(data_dir,'filenames_max20.pickle')
        with open(path,'rb') as f:
            data = pickle.load(f)
        return data
    
    def transfer_id_to_continuous(self,old_id):
        path = os.path.join(self.image_dir,'skip2conti.json')
        with open(path,'r') as f:
            dic = json.load(f)
        new_id = dic[str(old_id)]
        return new_id
    

    def __getitem__(self,index):
        image_name = self.filenames[index]
        clss = int(image_name.split('/')[0])
        class_id = self.transfer_id_to_continuous(clss)
        image_path = os.path.join(self.image_dir,image_name)
        imgs = get_imgs(image_path,transform=self.transform,normalize=self.norm) 
        
        return imgs, class_id

    def __len__(self):
        return len(self.filenames)

# dataloader for word embedding network
class WE_Data(data.DataLoader):
    
    def __init__(self, data_path,args,split='train'):
        self.args = args
        self.split = split
        aduio_path = os.path.join(data_path,'audio')
        if split=='train':
            self.audio_dir = os.path.join(aduio_path,'train2014')
        else:
            self.audio_dir = os.path.join(aduio_path,'val2014')            
        self.filenames = self.load_filelnames(self.audio_dir)

    def load_filelnames(self,data_dir):
        if self.split == 'train':
            path = data_dir + '/filenames/' + 'Bruce_1.0_None.json'
        else:
            path = data_dir + '/filenames/' + 'Bruce_1.0_None_uniqueImgID_with_categories.json'
        with open(path,'rb') as f:
            data = json.load(f)
        return data

    def load_json(self,path):
        with open(path,'rb') as f:
            data = json.load(f)
        return data

    def load_target(self,wav_duration,mel_len,timecode):
        target = np.zeros(mel_len)
        times = []
        for item in timecode:
            if 'WORD' in item:
                time = item[0]
                times.append(time)
        times = np.array(times)

        positions = times * mel_len / (wav_duration*1000)
        positions = np.around(positions).astype(np.int32)
        target[positions] = 1
        
        return target
    

    def __getitem__(self,index):
        data_dict = self.filenames[index]
        wav_name = data_dict['wavFilename']
        mel_name = wav_name.replace('.wav','.npy')
        mel_path = self.audio_dir + '/mel/' + mel_name 
        mel =  np.load(mel_path,allow_pickle=True)
        mel = (mel-mel.min())/(1.0-mel.min())
        
        json_path = self.audio_dir + '/json/' + wav_name.replace('.wav','.json')
        json_dict = self.load_json(json_path)

        wav_duration = json_dict['duration']
        mel_len = mel.shape[1]
        timecode = json_dict['timecode']
        mask = self.load_target(wav_duration,mel_len,timecode)

        return mel, mask, mel_len
        

    def __len__(self):
        return len(self.filenames)
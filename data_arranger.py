
# coding=UTF-8
# This Python file uses the following encoding: utf-8

import numpy as np
import os
from tqdm import tqdm
import cv2
from natsort import natsorted
import re
from sklearn.utils import shuffle
import pickle

#game='T1 vs DWG - Game 1 - Week 1 Day 1 S10 LCK Spring 2020 - T1 vs DAMWON Gaming G1 W1D1'
#game_path = 'extracted_games/'+game+'/'
def get_training_data(game_path, game):
    def get_mels(frame):
        path = mels_dir + 'mels_frame' + str(frame)+ '.jpg'
        img = cv2.imread(path) 
        img = img/255
        #img = img[:,:,np.newaxis]
        return img
    
    def get_frame_number(img_name):#讀取檔案名稱的frame number
        matchObj = re.match( r'.*_frame(.*).jpg', img_name)
        frame_number=matchObj.group(1)
        return int(frame_number)
    
    def properly_HL_img(img):#image的前處理
        path = HL_dir+'/'+img
        img = cv2.imread(path) 
        img = cv2.resize(img, (256, 144)) 
        img = img/255
        return img
    
    def properly_non_HL_img(img):#image的前處理
        path = non_HL_dir+'/'+img
        img = cv2.imread(path) 
        img = cv2.resize(img, (256, 144)) 
        img = img/255
        return img
    
    HL_dir = game_path +'samples/HL_from_full'
    HL_img_fileName_list = natsorted(os.listdir(HL_dir))
    
    non_HL_dir=game_path +'samples/Full_frames'
    non_HL_img_fileName_list = natsorted(os.listdir(non_HL_dir))
    
    mels_dir = game_path + 'mels/'

    
    img_list=[]#[img_batch,img_batch,img_batch,...]
    img_bunch=[]#[img,img,img,img,....]
    label_list=[]#[0,0,0,0,1,1,1,1,0,0,0,....]
    mels_list=[]#[mels,mels,mels,...]
    
    bunch_size=6
    
    #HL打包 
    for i in tqdm(range(len(HL_img_fileName_list)-bunch_size)):
        img_start = HL_img_fileName_list[i]
        img_end = HL_img_fileName_list[i + bunch_size]
        
        start_frame_number = get_frame_number(img_start)
        end_frame_number = get_frame_number(img_end)
        
        if start_frame_number + bunch_size == end_frame_number:#正確的一綑
            for j in range(bunch_size):
                img_bunch.append(properly_HL_img(HL_img_fileName_list[i+j]))
            img_list.append(img_bunch)
            img_bunch=[]
            label_list.append(1)            
            mels_list.append(get_mels(start_frame_number))
            #print(end_frame_number)
            
            
    non_HL_sample_count = len(img_list)#non_HL抽樣數量
    print('HL樣本數:'+str(non_HL_sample_count))
    
    #non_HL打包       
    for i in shuffle(range(len(non_HL_img_fileName_list)-bunch_size)):
        img_start = non_HL_img_fileName_list[i]
        img_end = non_HL_img_fileName_list[i + bunch_size]
        
        start_frame_number = get_frame_number(img_start)
        end_frame_number = get_frame_number(img_end)
        
        if start_frame_number + bunch_size == end_frame_number:#正確的一綑
            for j in range(bunch_size):
                img_bunch.append(properly_non_HL_img(non_HL_img_fileName_list[i+j]))
            img_list.append(img_bunch)
            img_bunch=[]
            label_list.append(0)            
            mels_list.append(get_mels(start_frame_number))
                      
            print('\r%d'%(non_HL_sample_count) , end='')
            non_HL_sample_count-=1
        if non_HL_sample_count == 0:#取得與HL相同數量時
            break
 
    
    images=np.array(img_list,dtype='float16')    
    mels=np.array(mels_list,dtype='float16')
    labels=np.array(label_list,dtype='float16')
    
    images, mels , labels = shuffle(images, mels, labels)
    return images, mels, labels



def save_training_data(game_path, game, save_dir):
    def get_mels(frame):
        path = mels_dir + 'mels_frame' + str(frame)+ '.jpg'
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img = img/255
        img = img[:,:,np.newaxis]
        img = np.array(img, dtype='float16')
        return img
    
    def get_frame_number(img_name):#讀取檔案名稱的frame number
        matchObj = re.match( r'.*_frame(.*).jpg', img_name)
        frame_number=matchObj.group(1)
        return int(frame_number)
    
    def properly_HL_img(img):#image的前處理
        path = HL_dir+'/'+img
        img = cv2.imread(path) 
        img = cv2.resize(img, (192, 108)) 
        img = img/255
        img = np.array(img, dtype='float16')
        return img
    
    def properly_non_HL_img(img):#image的前處理
        path = non_HL_dir+'/'+img
        img = cv2.imread(path) 
        img = cv2.resize(img, (192, 108)) 
        img = img/255
        img = np.array(img, dtype='float16')
        return img
    
    HL_dir = game_path +'samples/HL_from_full'
    HL_img_fileName_list = natsorted(os.listdir(HL_dir))
    
    non_HL_dir=game_path +'samples/Full_frames'
    non_HL_img_fileName_list = natsorted(os.listdir(non_HL_dir))
    
    mels_dir = game_path + 'mels_8/'

    
    img_list=[]#[img_batch,img_batch,img_batch,...]
    img_bunch=[]#[img,img,img,img,....]
    #label_list=[]#[0,0,0,0,1,1,1,1,0,0,0,....]
    mels_list=[]#[mels,mels,mels,...]
    
    bunch_size=8
    
    #HL打包 
    for i in range(len(HL_img_fileName_list)-bunch_size):
        img_start = HL_img_fileName_list[i]
        img_end = HL_img_fileName_list[i + bunch_size]
        
        start_frame_number = get_frame_number(img_start)
        end_frame_number = get_frame_number(img_end)
        
        if start_frame_number + bunch_size == end_frame_number:#正確的一綑
            for j in range(bunch_size):
                if j%2==0:
                    img_bunch.append(properly_HL_img(HL_img_fileName_list[i+j]))
            img_list.append(img_bunch)
            img_bunch=[]
            #label_list.append(1)            
            mels_list.append(get_mels(start_frame_number))
            #print(end_frame_number)
            
            
    non_HL_sample_count = len(img_list)#non_HL抽樣數量
    print('HL樣本數:'+str(non_HL_sample_count))
    
    #non_HL打包       
    for i in shuffle(range(len(non_HL_img_fileName_list)-bunch_size)):
        img_start = non_HL_img_fileName_list[i]
        img_end = non_HL_img_fileName_list[i + bunch_size]
        
        start_frame_number = get_frame_number(img_start)
        end_frame_number = get_frame_number(img_end)
        
        if start_frame_number + bunch_size == end_frame_number:#正確的一綑
            for j in range(bunch_size):
                if j%2==0:
                    img_bunch.append(properly_non_HL_img(non_HL_img_fileName_list[i+j]))
            img_list.append(img_bunch)
            img_bunch=[]
            #label_list.append(0)
            mels_list.append(get_mels(start_frame_number))
                      
            print('\r%d'%(non_HL_sample_count) , end='')
            non_HL_sample_count-=1
        if non_HL_sample_count == 0:#取得與HL相同數量時
            break

    if not os.path.exists(save_dir):#建立新的資料夾
        os.makedirs(save_dir)

    non_HL_sample_count = len(img_list)
    for i in range(len(img_list)):
        if i < non_HL_sample_count/2: #len(img_list)/2
            with open(save_dir +'HL_' + game + '_' + str(i), "wb") as f:   #Pickling
                pickle.dump([img_list[i],mels_list[i]], f)
        else:
            with open(save_dir +'NHL_' + game + '_' + str(i), "wb") as f:   #Pickling
                pickle.dump([img_list[i],mels_list[i]], f)
    print('saved.')


# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:09:51 2020

@author: Max
"""

import os
import cv2
import math
#from tqdm import tqdm

import moviepy.editor as mpe
#從影片萃取出frames 與audio (以秒為單位)
#game = 哪一場比賽 video_name=比賽的哪一部影片(FULL或HL)
def convert(game, video_name):
    print('啟動'+game)
    
    simpleName=video_name[0:20]+'_tmp'
    frames_path='extracted_games/'+ game +'/'+ simpleName +'/'
    
    #if not os.path.exists(audio_path):#建立新的資料夾
        #os.makedirs(audio_path)
    if not os.path.exists(frames_path):#建立新的資料夾
        os.makedirs(frames_path)
 
    video = mpe.VideoFileClip('paired_videos/'+ game +'/'+ video_name)
    

    for f in range(int(video.duration)):
        #cv2.imwrite(frames_path + simpleName + '_frame'  +str(f) + '.jpg', video.get_frame(f))
        video.save_frame(frames_path + simpleName + '_frame'  +str(f) + '.jpg', t=f)
        #print('\r%d'%(f) , end='')
    #for f in tqdm(range(int(video.duration))):
        #clip =video.subclip(f,f+1)
        #clip.audio.write_audiofile(audio_path + simple_video_name + '_frame' + str(f) + '.mp3', fps=22050, verbose=False, logger=None)
        #clip.save_frame(frames_path + simpleName + '_frame'  +str(f) + '.jpg', t=0)

    video.close()
    print('完成'+game)



#使用cv2取frames
#gameTitle = 哪一場比賽 video_name=比賽的哪一部影片(FULL或HL)
def Video2Frame(game, video_name):
    print('啟動'+game)
    
    simpleName=video_name[0:20]+'_tmp'
    frames_path='extracted_games/'+ game +'/'+ simpleName +'/'

    if not os.path.exists(frames_path):#建立新的資料夾
        os.makedirs(frames_path)
        
    cap = cv2.VideoCapture('paired_videos/'+game + '/' + video_name)
    frameRate = cap.get(5) #frame rate
    count = 0

    print('開始提取frames>>')
    
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename =frames_path+simpleName+"_frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
            print('\r%d'%(count) , end='')
    cap.release()
    #刪除最後一張圖片 因為音訊的部分會不滿1秒
    os.remove(frames_path + simpleName +'_frame'+ str(count-1) +'.jpg')
    
    print('提取結束')
    
    
    
#使用cv2取frames並壓縮(直接預測用)
def Video2Frame_resize(gameTitle,target_size=(256,144)):
    print('啟動'+gameTitle)
    simpleName=gameTitle[0:20]+'_resize'
    
    frames_path='output/'+simpleName+'/'

    if not os.path.exists(frames_path):#建立新的資料夾
        os.makedirs(frames_path)
        
    cap = cv2.VideoCapture('paired_videos/'+gameTitle)
    frameRate = cap.get(5) #frame rate
    count = 0
  
    print('開始提取frames>>')
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename =frames_path+simpleName+"_frame%d.jpg" % count;count+=1
            frame=cv2.resize(frame, target_size) 
            print('\r%d'%(count) , end='')
            cv2.imwrite(filename, frame)
    cap.release()
    print('提取結束')
    
    
"""
#從影片萃取出frames 與audio (以秒為單位)
#gameTitle = 哪一場比賽 video_name=比賽的哪一部影片(FULL或HL)
def convert(gameTitle, video_name):
    print('啟動'+gameTitle)
    
    simpleName=video_name[0:20]+'_tmp'
    frames_path='extracted_games/'+ gameTitle +'/'+ simpleName +'/'
    
    #if not os.path.exists(audio_path):#建立新的資料夾
        #os.makedirs(audio_path)
    if not os.path.exists(frames_path):#建立新的資料夾
        os.makedirs(frames_path)
 
    video = mp.VideoFileClip('paired_videos/'+gameTitle)
    
    if len(os.listdir(frames_path))!=int(video.duration):#確認是否已經輸出過
        for f in tqdm(range(int(video.duration))):
            clip =video.subclip(f,f+1)
            #clip.audio.write_audiofile(audio_path + simple_video_name + '_frame' + str(f) + '.mp3', fps=22050, verbose=False, logger=None)
            clip.save_frame(frames_path + simpleName + '_frame'  +str(f) + '.jpg', t=0)

    video.close()
    print('完成'+gameTitle)
"""      
        
        
        
        
        
        
        
        
        
        
        
        
        
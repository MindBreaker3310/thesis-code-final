# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:44:05 2019

@author: Max
"""


import pandas as pd
import os
import shutil
from natsort import natsorted

def rename_folders_to_Full_and_HL(game_path):
    #把frame多的資料夾改名成full_frames 少的改名成HL_frames
    folders=[]
    for file in os.listdir(game_path):
        if file[-4:]=='_tmp':
            folders.append(file)
    folder_1=folders[0]
    folder_2=folders[1]
    if len(os.listdir(game_path + folder_1))>len(os.listdir(game_path + folder_2)):
        os.rename(game_path + folder_1, game_path + 'Full_frames')
        os.rename(game_path + folder_2, game_path + 'HL_frames')
    else:
        os.rename(game_path + folder_2, game_path + 'Full_frames')
        os.rename(game_path + folder_1, game_path + 'HL_frames')


def extract_to_samples_folder(game_path):
    #依照abeled.csv 從Full_frames分出屬於HL的部分
    labeled_df=pd.read_csv(game_path + 'labeled.csv')   
    if not os.path.exists(game_path + 'HL_from_full'):#建立新的資料夾
        os.makedirs(game_path + 'HL_from_full')
    else:
        print('已存在HL_from_full')
    
    counter=0
    for frame in natsorted(os.listdir(game_path + 'Full_frames')):
        if (labeled_df.iloc[counter] == 1).any():#是HL
            shutil.move(game_path + 'Full_frames/'+frame, game_path + 'HL_from_full/'+frame)
        counter+=1
        
    
    #移至sample資料夾
    if not os.path.exists(game_path + 'samples'):#建立新的資料夾
        os.makedirs(game_path + 'samples')
    else:
        print('已存在sample')
    
    shutil.move(game_path + 'HL_from_full', game_path + 'samples')
    shutil.move(game_path + 'Full_frames', game_path + 'samples')


#把完成的影片與檔案放入finish_data資料夾下
#清空paired_videos/ 與 extracted_games/
def clearn_up_folders(arena_dir):
    arena_finish_path = 'finish_data/'+ arena_dir
    
    if not os.path.exists(arena_finish_path):#建立新的資料夾
        os.makedirs(arena_finish_path)
    
    #把所有配對的影片分別放到 finish_data/arena_dir/paired_videos/
    for paired_game in os.listdir('paired_videos/'):
        shutil.move('paired_videos/'+ paired_game +'/', arena_finish_path + 'paired_videos/' + paired_game)

    #把extracted_games/ 下的檔案移至 finish_data/arena_dir/extracted_games/
    for trained_game in os.listdir('extracted_games/'):
        shutil.move('extracted_games/'+ trained_game, arena_finish_path + 'extracted_games/' + trained_game)
















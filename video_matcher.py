# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:34:04 2020

@author: Max
"""



import re
import math
from collections import Counter
import os
import shutil

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = re.compile(r'(\w+)').findall(text)
    purify_words=[words[-1]]
    for i in range(len(words)-2,0,-1):
        if len(words[i]) == 1 and words[i].isdigit(): #是單獨的數字 'game', '2','day', '3' -> 'game2','day3'
            words[i-1] = words[i-1] + words[i]
        else:
            purify_words.append(words[i])
    purify_words.append(words[0])
    return Counter(purify_words)

def get_similarity(text1, text2):
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)
    
    cosine = get_cosine(vector1, vector2)
    return cosine


def move_paried_videos(threshold=0.78):
    if not os.path.exists('paired_videos/'):#建立新的資料夾
        os.makedirs('paired_videos/')
    for HL_video in os.listdir('HL_videos/'):
        max_similarity=0
        max_similarity_video = ''
        for full_video in os.listdir('full_videos/'):
            similarity = get_similarity(HL_video, full_video)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_video = full_video
        if max_similarity>=threshold:
            print('================')
            print(HL_video)
            print(max_similarity_video)
            print('similarity:'+ str(max_similarity))
            print('================')
            
            
            #只取比賽名稱 去掉下載的後輟與副檔名 .mp4
            paired_game_name = max_similarity_video[:-4]
            
            
            
            if not os.path.exists('paired_videos/'+ paired_game_name +'/'):#建立新的資料夾
                os.makedirs('paired_videos/'+ paired_game_name +'/')
            shutil.move('HL_videos/'+HL_video,'paired_videos/'+ paired_game_name +'/'+ HL_video)
            shutil.move('full_videos/'+max_similarity_video,'paired_videos/'+ paired_game_name +'/'+ max_similarity_video)



def pair_success_numbers(threshold=0.78):
    pair_success=0
    for HL_video in os.listdir('HL_videos/'):
        max_similarity=0
        max_similarity_video = ''
        for full_video in os.listdir('full_videos/'):
            similarity = get_similarity(HL_video, full_video)
            if similarity >= max_similarity:
                max_similarity = similarity
                max_similarity_video = full_video
        if max_similarity>=threshold:
            pair_success+=1
            print('================')
            print(HL_video)
            print(max_similarity_video)
            print('similarity:'+ str(max_similarity))
            print('================')
    return print( '配對數量:' + str(pair_success))





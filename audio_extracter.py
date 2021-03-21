# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 00:50:05 2020

@author: Max
"""
import librosa
import cv2
import numpy as np
import os
#from tqdm import tqdm


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def save_spectrogram_image(y, sr, out):
    # use melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mels = librosa.power_to_db(mels, ref=np.max)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save img
    cv2.imwrite(out, img)
    
#game = 賽事名稱 size = 6 = batch_size
def save_spectrograme(game, size):
    print('extracting audio..'+ game)
    
    output_path = 'extracted_games/'+ game + '/mels/'
    if not os.path.exists(output_path):#建立新的資料夾
        os.makedirs(output_path)
        
    full_video = 'paired_videos/'+ game +'/'+ game + '.mp4'
    
    
    duration = int(librosa.get_duration(filename=full_video))
    
    y, sr = librosa.load(full_video, duration = duration)
        
    for f in range(duration-size):
        save_spectrogram_image(y=y[f*sr:(f+size)*sr], sr=sr, out = output_path + 'mels_frame'+ str(f) +'.jpg')

    print('Done..')




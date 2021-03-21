# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:45:05 2020

@author: Max
"""




import cv2
from PIL import Image
import numpy as np
import pandas as pd
import joblib

classifier = joblib.load('digits_model_mlp_v3.pkl')
#classifier = joblib.load('digits_model_rf.pkl')  

#調整大小並不會破壞比例
def extend_resize(img):
    # read image
    img = np.stack((img,)*3, axis=-1)
    ht, wd, cc= img.shape    
    # create new image of desired size and color (black) for padding
    ww = 9
    hh = 13
    color = (0,0,0)
    result = np.full((hh,ww,cc), color, dtype=np.uint8)
    
    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2
    
    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = img
    
    result_img = Image.fromarray(result.astype('uint8'), 'RGB')
    
    result_img = [pixel/255.0 for pixel in iter(result_img.convert('L').getdata())]
    
    # save result
    return result_img


#抓出數字的輪廓
def get_contours_numbers(img):

    #img = Image.open(img_path)
    
    x, y, w, h= 945, 78, 43, 16
    
    crop_img = img.crop((x, y, x+w, y+h))
    #crop_img.show()
     
    np_img = np.array(crop_img)
    
    gray_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 100,255, cv2.THRESH_BINARY)
    #image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = sorted([(c, cv2.boundingRect(c)[0])for c in contours], key=lambda x:x[1])
    
    nice_contours=[]
    
    for (c,_) in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        #print((x,y,w,h))
        if 13>=h>=10 and 9>=w>=4:  
            nice_contours.append(image[y:y+h, x:x+w])
        #nice_contours.append((x,y,w,h))
        
    nice_numbers=[]  
    if len(nice_contours)==4:
        for img in nice_contours:
            nice_numbers.append(extend_resize(img))
            
    return nice_numbers


  
#predict    
def get_pred_numbers(img):
    nice_numbers = get_contours_numbers(img)

    if len(nice_numbers)==4:
        X=pd.DataFrame([nice_numbers[0],nice_numbers[1],nice_numbers[2],nice_numbers[3]])   
        pred_numbers=classifier.predict(X)    
        #print(pred_numbers)
        return pred_numbers
    else:
        return np.array([0,0,0,0],dtype='int64')

#img_path = 'C9 vs G2 Day 6 Group_frame127.jpg'
#get_pred_numbers(img_path)



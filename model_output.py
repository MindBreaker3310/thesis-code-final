from keras.models import load_model
import os
import cv2
import numpy as np
from tqdm import tqdm
import moviepy.editor as mp
from moviepy.editor import concatenate

#預設data/下有個是透過Video2Frame_resize所生成的資料夾 內含所有壓縮過的full video的frames
def pred_and_save_numpy(npy_saveName = 'pred_npy', model_name = '2d_conv_lstm_model.h5'):
    resized_folder=os.listdir('data/')[0]
    
    def get_frame(f):
        img = cv2.imread('data/'+resized_folder+'/'+ resized_folder + '_frame' + str(f) + '.jpg')
        img = cv2.resize(img, (256, 144)) 
        img=img/255
        return img

    model = load_model(model_name)
    model.summary()
    
    
    #silding window要移動的次數
    batch_size = 6
    moves = len(os.listdir('data/' + resized_folder +'/')) - batch_size + 1

    samples_list=[]
    img_batch=[]
    for i in tqdm(range(moves)):
        for j in range(batch_size):
            img_batch.append(get_frame(i+j))
        samples_list.append(img_batch)
        img_batch=[]

    X_samples=np.array(samples_list, dtype='float32') 
    y_pred = model.predict(X_samples)

    pred=[]
    for i in y_pred:
        if i[0]<0.5:#threshold
            pred.append(0)
        else:
            pred.append(1)            
            
    #如果沒有output資料夾就建立一個        
    if not os.path.exists('output/'):
        os.makedirs('output/')
        
    np.save('output/'+ npy_saveName +'.npy',pred)
    print('numpy array saved.')



#輸出要剪輯的區段
def composite_HL(npy_saveName = 'pred_npy'):
    y_pred=np.load('output/'+ npy_saveName +'.npy')
    #10000只是個大數字 無意義 若影片大於10000秒就搓屎
    HL_zone=[10000,0]
    HL_list=[]
    for i in range(len(y_pred)):
        if y_pred[i]==1:
            if HL_zone[1]<i and HL_zone[1]!=0:
                HL_list.append(HL_zone)
                HL_zone=[10000,0]
            if HL_zone[0]>i:           
                HL_zone[0]=i
            if HL_zone[1]<=i:
                HL_zone[1]=i+5
    HL_zone[1]+=5
    HL_list.append(HL_zone)
    return HL_list


#讓畫面更流暢
def purify_cuting_list(HL_list):
    purify_HL_list=[]
    current_HL=HL_list[0]
    for i in range(1,len(HL_list)):
        if HL_list[i][0]-current_HL[1]<=8:#與前一段差8秒
            current_HL[1]=HL_list[i][1]#就與下一段HL相接
        else:#獨立片段
            if current_HL[1]-current_HL[0]>8:#超過8秒精華才加入剪輯
                purify_HL_list.append(current_HL)
            current_HL=HL_list[i]
    purify_HL_list.append(current_HL)
    return purify_HL_list



#預設videos內只有一部影片 且是完整的賽事影片
def write_HL_video(mp4_saveName, HL_list):
    full_video=os.listdir('videos/')[0]
    video = mp.VideoFileClip('videos/'+full_video)

    clips=[]
    for HL in HL_list:
        clip =video.subclip(HL[0], HL[1])
        clips.append(clip)

    faded_clips = [clip.crossfadein(1) for clip in clips]
    final_clip = concatenate( faded_clips, padding=-1, method="compose")
    final_clip.write_videofile('output/'+ mp4_saveName +'.mp4', threads=12, fps=23.976) 
    video.close()
    


    



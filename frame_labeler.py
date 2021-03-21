
import os
from PIL import Image
import pandas as pd
from natsort import natsorted
#import digital_recognizer
import digital_recognizer_v2 as digital_recognizer


#====================================#
#      HL_video's Frame->HL_zone     #
#====================================#
#判斷這場比賽HL截圖所有的時間
def predict_HL_Frames(game_path):#game_path=哪一場比賽的資料夾  
    predictTime=[]
    for frame in natsorted(os.listdir(game_path + 'HL_frames')):#比賽資料夾下的HL_frames   
        #每一個frame截圖          
        img = Image.open(game_path + 'HL_frames/'+frame)
        #使用digits_model_mlp.pkl辨識秒數
        pred_numbers = digital_recognizer.get_pred_numbers(img)
        
        #換成秒數
        predictTime.append(int("{}{}".format(pred_numbers[0],pred_numbers[1]))*60+int("{}{}".format(pred_numbers[2],pred_numbers[3])))
        """
        #把預測錯誤的洞補起來 ex:  111,0,113 -> 111,112,113
        for i in range(1,len(predictTime)-1):
            if predictTime[i]==0 and predictTime[i-1]!=0 and predictTime[i+1]!=0:
                predictTime[i] = int((predictTime[i-1] + predictTime[i+1])/2)
        """
        
    return predictTime

#抓出連續的時間
def get_continuing_second(frame_list):
    temp_list = []
    highLight_list = []
    for i in range(len(frame_list)):
        if len(temp_list) == 0:
            temp_list.append(frame_list[i])
        else:
            change = frame_list[i] - temp_list[-1]
            if change <=4 and change >=0:#差0~2秒
                temp_list.append(frame_list[i])
            else:#差超過一秒，可能換畫面
                if len(temp_list)>=4:#連續2秒的畫面
                    highLight_list.append(temp_list)
                temp_list=[]
                temp_list.append(frame_list[i])
    return highLight_list

#抓出HL的開始與結束並存成csv
def get_HL_time_zone(game_path):
    print('frame2Time執行>>>')
    #判斷這場比賽HL截圖所有的時間
    predictTime_list=predict_HL_Frames(game_path)    
    #抓出連續的時間
    Continuing_time_list=get_continuing_second(predictTime_list)
    
    hl_start_and_end=[]
    for hl in Continuing_time_list:
        if hl[0]!=hl[-1]:
            hl_start_and_end.append([hl[0],hl[-1]])
            
    hl_start_and_end2=[] 
    skip_next=False
    for i in range(len(hl_start_and_end)-1):
        if skip_next:
            skip_next=False
            continue
        if hl_start_and_end[i+1][0] - hl_start_and_end[i][1] <= 4 and hl_start_and_end[i+1][0] - hl_start_and_end[i][1] >= 0:#下一個HL開頭 - 這一個HL的結尾 小於4秒的話
            hl_start_and_end2.append([hl_start_and_end[i][0],hl_start_and_end[i+1][1]]) #把兩個HL合併
            skip_next=True
        elif hl_start_and_end[i][1] - hl_start_and_end[i][0] >= 5:
            hl_start_and_end2.append(hl_start_and_end[i])
            
    if not skip_next and hl_start_and_end[-1][1] - hl_start_and_end[-1][0] >= 5:#處理最後一段HL
        hl_start_and_end2.append(hl_start_and_end[-1])
            
    pd.DataFrame(hl_start_and_end2).to_csv( game_path + 'HL_times.csv',index=False)
    print('DONE.....')         
#輸出HL_times.csv(單位:秒)
#HL_start     HL_end
#    456       491    
#    1123      1221    
#    ...       ... 

#==================================#
#  Full_video's Frame->Times list  #
#==================================#
#輸出每張frame的預測
def get_full_time_list(game_path):
    print('full_video_frame_time>>>>>start')
    
    full_frame_time_list=[]
    for frame in natsorted(os.listdir(game_path +'Full_frames')):  
        #讀取frame截圖
        img = Image.open(game_path +'Full_frames/'+frame)    
        #使用digits_model_mlp.pkl辨識秒數
        pred_numbers = digital_recognizer.get_pred_numbers(img)
        
        full_frame_time_list.append(int("{}{}".format(pred_numbers[0],pred_numbers[1]))*60+int("{}{}".format(pred_numbers[2],pred_numbers[3])))
    
    #把預測錯誤的洞補起來 ex:  111,0,113 -> 111,112,113
    for i in range(1,len(full_frame_time_list)-1):
        if full_frame_time_list[i]==0 and full_frame_time_list[i-1]!=0 and full_frame_time_list[i+1]!=0:
            full_frame_time_list[i] = int((full_frame_time_list[i-1] + full_frame_time_list[i+1])/2)
    
    time_list_df=pd.DataFrame(full_frame_time_list)
    time_list_df.columns=['times']
    time_list_df.to_csv(game_path +'Full_video_times.csv',index=False) 
    print('full_video_frame_time>>>>>Done.')
#輸出 Full_video_times.csv
#mlp辨識的秒數(單位:秒)
#    456  
#    457    
#    459
#    ...     


#==================================#
#     label full_video_time_list   #
#==================================#

def label_full_time_list(game_path):
    print('label_full_time_list>>>>>start')
    
    HL_zone_df=pd.read_csv(game_path +'HL_times.csv')#HL區間 AKA開頭與結尾    
    HL_zone_list=[] #[[HL_start,HL_end],[HL_start,HL_end],[HL_start,HL_end],....]
    for i in range(len(HL_zone_df)):
        HL_zone_list.append([HL_zone_df.iloc[i][0],HL_zone_df.iloc[i][1]])
    

    time_list_df=pd.read_csv(game_path +'Full_video_times.csv')#full_video的時間 [0,0,0,0,331,332,333,334,335...]
    time_list=time_list_df['times'].tolist()
    

    labeled_df=pd.read_csv(game_path +'Full_video_times.csv')#full_video的時間 [0,0,0,0,331,332,333,334,335...]
    labeled_df.columns=['hl']
    for i in range(len(labeled_df)):
        labeled_df.iloc[i]=0
    
    not_find = []
    for i in range(len(HL_zone_list)):    #第i段HL
        HL_start = HL_zone_list[i][0] #HL開始的比賽秒數
        HL_end = HL_zone_list[i][1] #HL結束的比賽秒數
        HL_length = HL_end-HL_start
        find = False
        #print(HL_start,HL_end,HL_length)
        
        for f in range(len(time_list) - HL_length ):#full_video的第f張frame 非比賽秒數
            #time_list[f]幀數時間=比賽秒數
            if time_list[f]==HL_start and time_list[f+HL_length]==HL_end: #HL有頭 有尾
                print('HL有頭 有尾')
                print(HL_start,HL_end,HL_length)                
                for j in range(f,f + HL_length +2):#標記為HL
                    labeled_df.iloc[j]=1   
                find = True  
                break
            if time_list[f]==HL_start and time_list[f+HL_length]!=HL_end: #HL 有頭 無尾
                print('HL 有頭 無尾',i)
                print(HL_start,HL_end,HL_length)
                print(time_list[f+HL_length])#應該要為HL_end的值 但不是
                
                correct_end_f = 0
                err_distance = 10
                for e in range(4):#正負3秒 找最小誤差
                    forward_e = abs(time_list[f + HL_length - e] - HL_end)
                    backward_e = abs(time_list[f + HL_length + e] - HL_end)
                    if backward_e <= err_distance:
                        correct_end_f = f + HL_length + e
                        err_distance = backward_e
                    if forward_e < err_distance:
                        correct_end_f = f + HL_length - e
                        err_distance = forward_e
                print(err_distance)
                if correct_end_f != 0 and err_distance <= 3:
                    for j in range(f,correct_end_f+1):#標記為HL
                        labeled_df.iloc[j]=1
                    print('find')
                    print(time_list[f],time_list[correct_end_f+1])            
                    find = True                                           
                    break           
            if time_list[f+HL_length]==HL_end-1 and time_list[f]!=HL_start: #HL 有尾 無頭
                print('HL 有尾 無頭',i)
                print(HL_start,HL_end,HL_length)
                print(time_list[f])#應該要為HL_start的
                
                correct_start_f = 0
                err_distance = 10
                for e in range(4):#正負3秒 找最小誤差
                    forward_e = abs(time_list[f - e] - HL_start)
                    backward_e = abs(time_list[f + e] - HL_start)
                    if backward_e <= err_distance:
                        correct_start_f = f + e
                        err_distance = backward_e
                    if forward_e < err_distance:
                        correct_start_f = f - e
                        err_distance = forward_e
                print(err_distance)
                if correct_start_f != 0 and err_distance <= 3:          
                    for j in range(correct_start_f,f+HL_length+2):#標記為HL
                        labeled_df.iloc[j]=1   
                    print('find')
                    #print(time_list[correct_start_f],time_list[f+HL_length])   
                    print(time_list[correct_start_f],time_list[f+HL_length+1])
                    find = True              
                    break

        if not find:
            not_find.append(i) 
               
    still_not_find=[]
    if len(not_find) != 0:#處理找不到的        
        for i in range(len(HL_zone_list)):    #第i段HL
            HL_start = HL_zone_list[i][0] #HL開始的比賽秒數
            HL_end = HL_zone_list[i][1] #HL結束的比賽秒數
            HL_length = HL_end-HL_start
            find = False
            #print(HL_start,HL_end,HL_length)
            
            for f in range(len(time_list) - HL_length ):#full_video的第f張frame 非比賽秒數
                #time_list[f]幀數時間=比賽秒數
                if time_list[f]==HL_start and time_list[f+HL_length]==HL_end: #HL有頭 有尾
                    #print('goood')
                    #print(HL_start,HL_end,HL_length)                
                    for j in range(f,f + HL_length +1):#標記為HL
                        labeled_df.iloc[j]=1   
                    find = True  
                    break
                if time_list[f]==HL_start: #HL 有頭 無尾
                    print('HL 有頭 無尾',i)
                    print(HL_start,HL_end,HL_length)
                    print(time_list[f+HL_length])#應該要為HL_end的值 但不是
                    
                    correct_end_f = 0
                    err_distance = 10
                    for e in range(4):#正負3秒 找最小誤差
                        forward_e = abs(time_list[f + HL_length - e] - HL_end)
                        backward_e = abs(time_list[f + HL_length + e] - HL_end)
                        if backward_e < err_distance:
                            correct_end_f = f + HL_length + e
                            err_distance = backward_e
                        if forward_e < err_distance:
                            correct_end_f = f + HL_length - e
                            err_distance = forward_e
                    print(err_distance)
                    if correct_end_f != 0 and err_distance <= 3:
                        for j in range(f,correct_end_f+1):#標記為HL
                            labeled_df.iloc[j]=1
                        print('find')
                        print(time_list[f],time_list[correct_end_f+1])            
                        find = True                                           
                        break           
                if time_list[f+HL_length]==HL_end-1: #HL 有尾 無頭
                    print('HL 有尾 無頭',i)
                    print(HL_start,HL_end,HL_length)
                    print(time_list[f])#應該要為HL_start的
                    
                    correct_start_f = 0
                    err_distance = 10
                    for e in range(4):#正負3秒 找最小誤差
                        forward_e = abs(time_list[f - e] - HL_start)
                        backward_e = abs(time_list[f + e] - HL_start)
                        if backward_e < err_distance:
                            correct_start_f = f + e
                            err_distance = backward_e
                        if forward_e < err_distance:
                            correct_start_f = f - e
                            err_distance = forward_e
                    print(err_distance)
                    if correct_start_f != 0 and err_distance <= 3:          
                        for j in range(correct_start_f,f+HL_length+1):#標記為HL
                            labeled_df.iloc[j]=1   
                        print('find')
                        #print(time_list[correct_start_f],time_list[f+HL_length])   
                        print(time_list[correct_start_f],time_list[f+HL_length+1])
                        find = True              
                        break      
        if not find:
            still_not_find.append(i)         
    
        
        
    if len(still_not_find)==0:
        print('完美')
    else:
        print('部分出錯')
        print(still_not_find)

    labeled_df = pd.concat([time_list_df,labeled_df], axis=1)
    labeled_df.to_csv(game_path +'labeled.csv',index=False)   

    print('label_full_time_list>>>>>Done.')
#輸出 labeled.csv
#mlp辨識的秒數(單位:秒)   依照HL_times.csv判斷是否為HL
#    456                           0
#    457                           1
#    459                           1
#    ...                           .
    
    
    

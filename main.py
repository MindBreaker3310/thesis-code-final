import os
import frame_extracter
import audio_extracter
import frame_labeler
import folder_toolbox
import data_arranger
from keras.models import load_model
import model_output
import video_matcher
from multiprocessing import Pool
import vs_model
import pickle
import keras.backend as K
import sys
import traceback

#=======================#
#     事前處理的部分     #
#=======================#
#print出所有會配對到的影片  人工檢查 確保不會出錯 並回傳共有幾場配對成功
video_matcher.pair_success_numbers(threshold=0.78)

#將所有配對成功的影片 放入paried_videos/內
video_matcher.move_paried_videos(threshold=0.78)

#game='UOL vs SPY - Game 2 - Knockouts Play-Ins S9 LoL Worlds 2019 - Unicorns of Love vs Splyce G2 [1080p]'
#video_name=os.listdir('paired_videos/'+ game)[0]

#===================================================
#以下multiprocessing程式碼必須在原生windows cmd內執行  
def multi_job1(game):
    try:
        #預期paired_videos/+game裡面只有兩部影片 1.完整的賽事影片 2.別人剪輯過的精華
        for video_name in os.listdir('paired_videos/'+ game):
            #使用cv2取frames 儲存到extracted_games/
            #frame_extracter.Video2Frame(game,video_name)
            #使用moviepy取frames
            frame_extracter.convert(game, video_name)
        
        #提取 mel frequency 每六秒存成一張 在extracted_games/game/mels/
        audio_extracter.save_spectrograme(game, 6)
    except Exception as e:
        print('====================')
        print(game)
        e = sys.exc_info()
        print('Error Return Type: ', type(e))
        print('Error Class: ', e[0])
        print('Error Message: ', e[1])
        print('Error Traceback: ', traceback.format_tb(e[2]))         
        
def multi_job2(game):
    try:
        #賽區需要統一設定 LCK=韓國  WR=2019世界賽 LEC=歐洲 LCS=北美
        #arena = 'WR'
    
        game_path = 'extracted_games/'+ game +'/'
        #把extracted_games/game/下的資料夾改名  檔案多的叫full_frames 少的改名成HL_frames
        folder_toolbox.rename_folders_to_Full_and_HL(game_path)
    
        #從擷取下來的frames取得賽是秒數 使用digits_model_mlp.pkl
        #LCK是韓國的賽事，其中的數字字形、位置與歐美的不同
        frame_labeler.get_HL_time_zone(game_path)#輸出HL_times.csv
        frame_labeler.get_full_time_list(game_path)#輸出 Full_video_times.csv
        frame_labeler.label_full_time_list(game_path)#輸出 labeled.csv
    
        #依照labeled.csv 從Full_frames分出屬於HL的部分 移至sample資料夾
        folder_toolbox.extract_to_samples_folder(game_path)
    except Exception as e:
        print('====================')
        print(game)
        e = sys.exc_info()
        print('Error Return Type: ', type(e))
        print('Error Class: ', e[0])
        print('Error Message: ', e[1])
        print('Error Traceback: ', traceback.format_tb(e[2]))    
if __name__=='__main__':
    pool = Pool(os.cpu_count())
    pool.map(multi_job1, os.listdir('paired_videos/'))
    pool.map(multi_job2, os.listdir('extracted_games/'))

#以上multiprocessing程式碼必須在原生windows cmd內執行  
#===================================================


#=======================#
#     更新模型的部分     #
#=======================#


#初始化 log file
#log=[]
#with open('trained_log', 'wb') as f:
#    pickle.dump(log, f)
    
#訓練所有提取完成的檔案
for game in os.listdir('extracted_games/'):
    print(game)
    game_path = 'extracted_games/'+ game +'/'        

    with open('trained_log', 'rb') as f:
        log = pickle.load(f)
    if game not in log:#如果沒訓練過的話
        print('training')
        #訓練model
        if not os.path.exists('vs_model.h5'):#first time training
            model = vs_model.get_model()
            images, mels, labels = data_arranger.get_training_data(game_path, game)
            model.summary()
            model.fit([images, mels], labels , epochs=15, validation_split=0.05, batch_size=16)
            model.save('vs_model.h5')
        else:
            images, mels, labels = data_arranger.get_training_data(game_path, game)
            model = load_model('vs_model.h5')
            model.fit([images, mels], labels , epochs=15, validation_split=0.05, batch_size=16)
            model.save('vs_model.h5')
            del model, images, mels, labels
            K.clear_session()
        log.append(game)
        with open('trained_log', 'wb') as f:
            pickle.dump(log, f)
    else:#已經訓練過了
        print('trained')
    

#清空paired_videos/ 與 extracted_games/
folder_toolbox.clearn_up_folders(arena = 'WR')













#下面還沒做修改!!!!


#NO.2
#一次輸出多部影片
#需要輪流將full_video放入videos/內

#=======================#
#     直接輸出的部分     #
#=======================#

#預期videos裡面只有一部影片 1.完整的賽事影片
for video in os.listdir('videos/'):
    #使用cv2取frames並壓縮(直接預測用)
    frame_extracter.Video2Frame_resize(video,target_size=(256,144))

#要儲存的檔名 任意
saveName=os.listdir('videos/')[0]
#儲存預測的list
model_output.pred_and_save_numpy(npy_saveName = 'pred_npy', model_name = '2d_conv_lstm_model.h5')
#輸出要剪輯的區段
#依照預測的list決定要剪輯的片段
pred_HL_list = model_output.composite_HL(npy_saveName = saveName)
#讓畫面更 去除掉過短的精華
purify_HL_list = model_output.purify_cuting_list(pred_HL_list)
#輸出影片
model_output.write_HL_video(saveName, purify_HL_list)




















        
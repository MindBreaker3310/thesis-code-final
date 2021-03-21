import os
import frame_extracter
import audio_extracter
import frame_labeler
import folder_toolbox
from multiprocessing import Pool
import sys
import traceback

#===================================================
#以下multiprocessing程式碼必須在原生windows cmd內執行  
def multi_job1(game):
    try:

        #預期paired_videos/+game裡面只有兩部影片 1.完整的賽事影片 2.別人剪輯過的精華
        for video_name in os.listdir('paired_videos/'+ game):
            #使用moviepy取frames
            frame_extracter.convert(game, video_name)
        
        #提取 mel frequency 每六秒存成一張 在extracted_games/game/mels/
        audio_extracter.save_spectrograme(game, 8)

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
        game_path = 'extracted_games/'+game+'/'
        #把extracted_games/game/下的資料夾改名  檔案多的叫full_frames 少的改名成HL_frames
        folder_toolbox.rename_folders_to_Full_and_HL(game_path)
    
        #從擷取下來的frames取得賽是秒數 使用digits_model_mlp.pkl
        frame_labeler.get_full_time_list(game_path)#輸出 Full_video_times.csv
        frame_labeler.get_HL_time_zone(game_path)#輸出HL_times.csv        
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

import data_arranger
def multi_job3(game):
    print(game)
    game_path = 'extracted_games/'+ game +'/'        
    data_arranger.save_training_data(game_path, game, save_dir='data_set/')

    
if __name__=='__main__':
    pool = Pool(os.cpu_count())   
    #pool.map(multi_job1, os.listdir('paired_videos/'))
    #pool.map(multi_job2, os.listdir('extracted_games/'))

    #save training data to samples folder
    pool = Pool(2)
    pool.map(multi_job3, os.listdir('extracted_games/'))

    #folder_toolbox.clearn_up_folders(arena_dir='LCK_spring_2020/')

#以上multiprocessing程式碼必須在原生windows cmd內執行  
#===================================================

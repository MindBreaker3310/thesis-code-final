import os
import folder_toolbox
import data_arranger
from keras.models import load_model
import vs_model
import pickle
import keras.backend as K


#=======================#
#     更新模型的部分     #
#=======================#


#初始化 log file
#log=[]
#with open('trained_log', 'wb') as f:
#    pickle.dump(log, f)
"""
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
            model.fit([images, mels], labels , epochs=15, validation_split=0.05, batch_size=12)
            model.save('vs_model.h5')
            del model, images, mels, labels
            K.clear_session()
        else:
            images, mels, labels = data_arranger.get_training_data(game_path, game)
            model = load_model('vs_model.h5')
            model.fit([images, mels], labels , epochs=15, validation_split=0.05, batch_size=12)
            model.save('vs_model.h5')
            del model, images, mels, labels
            K.clear_session()
        log.append(game)
        with open('trained_log', 'wb') as f:
            pickle.dump(log, f)
    else:#已經訓練過了
        print('trained')
"""

#清空paired_videos/ 與 extracted_games/
#folder_toolbox.clearn_up_folders(arena = 'VAL')







        

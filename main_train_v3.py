import os
import numpy as np
import random
import pickle
import vs_model
import data_arranger
"""
def generate_data(directory, batch_size):
    #Replaces Keras' native ImageDataGenerator.
    i = 0
    file_list = os.listdir(directory)
    while True:
        image_batch = []
        mel_batch = []
        labels_batch = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                random.shuffle(file_list)
            
            sample = file_list[i]
            with open(directory + sample, "rb") as f:   # Unpickling
                train_data = pickle.load(f)
                image_batch.append(train_data[0])
                mel_batch.append(train_data[1])
            
            if sample[:2] =='HL':
                labels_batch.append(1)
            else:
                labels_batch.append(0)
            i += 1
        yield ([np.array(image_batch), np.array(mel_batch)], np.array(labels_batch))


model = vs_model.get_model()
batch_size = 12

from keras.callbacks import EarlyStopping, ModelCheckpoint
model_callbacks = [EarlyStopping(monitor='val_loss', patience=13, mode='min'), ModelCheckpoint('vs_model.h5', monitor='val_loss', save_best_only=True, mode='min')]

history = model.fit_generator(generator = generate_data('samples_train/', batch_size), steps_per_epoch = len(os.listdir('samples_train/')) // batch_size, validation_data = generate_data('samples_val/', batch_size), validation_steps = len(os.listdir('samples_val/')) // batch_size, epochs = 50, callbacks = model_callbacks)

with open('trainHistoryDict', 'wb') as f:
    pickle.dump(history.history, f)




#image only
def generate_data(directory, batch_size):
    #Replaces Keras' native ImageDataGenerator.
    i = 0
    file_list = os.listdir(directory)
    while True:
        image_batch = []
        #mel_batch = []
        labels_batch = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                random.shuffle(file_list)
            
            sample = file_list[i]
            with open(directory + sample, "rb") as f:   # Unpickling
                train_data = pickle.load(f)
                image_batch.append([train_data[0][0], train_data[0][2], train_data[0][4]])
                #mel_batch.append(train_data[1])
            
            if sample[:2] =='HL':
                labels_batch.append(1)
            else:
                labels_batch.append(0)
            i += 1
        yield (np.array(image_batch), np.array(labels_batch))


model = vs_model.get_image_model()
batch_size = 12

from keras.callbacks import EarlyStopping, ModelCheckpoint
model_callbacks = [EarlyStopping(monitor='val_loss', patience=5, mode='min'), ModelCheckpoint('vs_image_model_v2.h5', monitor='val_loss', save_best_only=True, mode='min')]

history = model.fit_generator(generator = generate_data('samples_train/', batch_size), steps_per_epoch = len(os.listdir('samples_train/')) // batch_size, validation_data = generate_data('samples_val/', batch_size), validation_steps = len(os.listdir('samples_val/')) // batch_size, epochs = 50, callbacks = model_callbacks)

with open('trainHistoryDict_image_v2', 'wb') as f:
    pickle.dump(history.history, f)



#audio only
def generate_data(directory, batch_size):
    #Replaces Keras' native ImageDataGenerator.
    i = 0
    file_list = os.listdir(directory)
    while True:
        #image_batch = []
        mel_batch = []
        labels_batch = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                random.shuffle(file_list)
            
            sample = file_list[i]
            with open(directory + sample, "rb") as f:   # Unpickling
                train_data = pickle.load(f)
                #image_batch.append(train_data[0])
                mel_batch.append(train_data[1])
            
            if sample[:2] =='HL':
                labels_batch.append(1)
            else:
                labels_batch.append(0)
            i += 1
        yield (np.array(mel_batch), np.array(labels_batch))


model = vs_model.get_audio_model()
batch_size = 12

from keras.callbacks import EarlyStopping, ModelCheckpoint
model_callbacks = [EarlyStopping(monitor='val_loss', patience=5, mode='min'), ModelCheckpoint('vs_audio_model_v2.h5', monitor='val_loss', save_best_only=True, mode='min')]

history = model.fit_generator(generator = generate_data('train_samples/', batch_size), steps_per_epoch = len(os.listdir('train_samples/')) // batch_size, validation_data = generate_data('val_samples/', batch_size), validation_steps = len(os.listdir('val_samples/')) // batch_size, epochs = 50, callbacks = model_callbacks)

with open('trainHistoryDict_audio_v2', 'wb') as f:
    pickle.dump(history.history, f)


def generate_data(data_set, batch_size):
    #Replaces Keras' native ImageDataGenerator.
    directory = 'data_set/'
    i = 0
    file_list = data_set
    while True:
        image_batch = []
        mel_batch = []
        labels_batch = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                random.shuffle(file_list)
            
            sample = file_list[i]
            with open(directory + sample, "rb") as f:   # Unpickling
                train_data = pickle.load(f)
                #image_batch.append([train_data[0][1], train_data[0][3], train_data[0][5], train_data[0][7]])
                image_batch.append([train_data[0][0], train_data[0][1], train_data[0][2], train_data[0][3]])
                mel_batch.append(train_data[1])
            
            if sample[:2] =='HL':
                labels_batch.append(1)
            else:
                labels_batch.append(0)
            i += 1
        #yield (np.array(image_batch), np.array(labels_batch))
        #yield (np.array(mel_batch), np.array(labels_batch))
        yield ([np.array(image_batch), np.array(mel_batch)], np.array(labels_batch))




from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold



data_set = os.listdir('data_set/')


i=0
for train_index,test_index in KFold(n_splits=5, shuffle=True, random_state=0).split(data_set):#把samples_list分成5份 4份為train 1份為test 並循環5次
    #print(train_index, test_index)
    train_list = np.array(data_set)[train_index]
    test_list = np.array(data_set)[test_index]
    
    #model = vs_model.get_2d_lstm_model_image_only()
    #model = vs_model.get_3d_cnn_model()
    #model = vs_model.get_2d_lstm_model_8()
    model = vs_model.get_3d_cnn_model_image_audio()
    batch_size = 32

    model_callbacks = [EarlyStopping(monitor='val_loss', patience=5, mode='min'), ModelCheckpoint('vs_model_'+ str(i) +'.h5', monitor='val_loss', save_best_only=True, mode='min')] 
    history = model.fit_generator(generator = generate_data(train_list, batch_size), steps_per_epoch = len(train_list) // batch_size, validation_data = generate_data(test_list, batch_size), validation_steps = len(test_list) // batch_size, epochs = 50, callbacks = model_callbacks)

    with open('trainHistoryDict_'+ str(i), 'wb') as f:
        pickle.dump(history.history, f)
    i+=1

"""

#for simple cnn

def generate_data(data_set, batch_size):
    #Replaces Keras' native ImageDataGenerator.
    directory = 'data_set/'
    i = 0
    file_list = data_set
    while True:
        image_batch = []
        mel_batch = []
        labels_batch = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                random.shuffle(file_list)
            
            sample = file_list[i]
            with open(directory + sample, "rb") as f:   # Unpickling
                train_data = pickle.load(f)
                image_batch.append(train_data[0][0])
                image_batch.append(train_data[0][1])
                image_batch.append(train_data[0][2])
                image_batch.append(train_data[0][3])
                #mel_batch.append(train_data[1])

            
            if sample[:2] =='HL':
                labels_batch.append(1)
                labels_batch.append(1)
                labels_batch.append(1)
                labels_batch.append(1)
            else:
                labels_batch.append(0)
                labels_batch.append(0)
                labels_batch.append(0)
                labels_batch.append(0)
            i += 1
        yield (np.array(image_batch), np.array(labels_batch))
        #yield (np.array(mel_batch), np.array(labels_batch))
        #yield ([np.array(image_batch), np.array(mel_batch)], np.array(labels_batch))






from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from keras.models import load_model
from tqdm import tqdm

data_set = os.listdir('data_set/')


i=0
for train_index,test_index in KFold(n_splits=5, shuffle=True, random_state=0).split(data_set):#把samples_list分成5份 4份為train 1份為test 並循環5次
    #print(train_index, test_index)
    train_list = np.array(data_set)[train_index]
    test_list = np.array(data_set)[test_index]

    #model = vs_model.get_2d_lstm_model_image_only()
    #model = vs_model.get_3d_cnn_model_image_noly()
    model = vs_model.get_simple_cnn_model_image_only()
    batch_size = 32

    model_callbacks = [EarlyStopping(monitor='val_loss', patience=5, mode='min'), ModelCheckpoint('vs_model_'+ str(i) +'.h5', monitor='val_loss', save_best_only=True, mode='min')] 
    history = model.fit_generator(generator = generate_data(train_list, batch_size), steps_per_epoch = len(train_list) // batch_size, validation_data = generate_data(test_list, batch_size), validation_steps = len(test_list) // batch_size, epochs = 50, callbacks = model_callbacks)


    #test prediction
    model = load_model('vs_model_'+ str(i) +'.h5')
    directory = 'data_set/'
    true_count=0
    false_count=0
    
    for sample in tqdm(test_list):
        pred_sum=0
        with open(directory + sample, "rb") as f:   # Unpickling
            train_data = pickle.load(f)
            for j in range(4):
                pred = model.predict(np.expand_dims(train_data[0][j], axis=0))
                pred_sum += pred
        
        if (pred_sum/4 >= 0.5 and sample[:2] =='HL') or (pred_sum/4 < 0.5 and sample[:2] =='NH'):
            #預測正確
            true_count+=1
        else:
            #預測錯誤
            false_count+=1
    print(true_count/(true_count + false_count))
    with open('trainHistoryDict_'+ str(i), 'wb') as f:
        pickle.dump([history.history, true_count/(true_count + false_count)], f)
    i+=1



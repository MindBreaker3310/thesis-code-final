from keras.layers import Input, TimeDistributed, LSTM, Bidirectional, Concatenate, Activation, BatchNormalization, Convolution3D
from keras.layers.convolutional import  MaxPooling3D
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.applications.mobilenet import MobileNet
from keras.utils import plot_model, print_summary


def get_2d_lstm_model():

    input_A = Input(shape=(None,108,192,3))
    
    image_model = TimeDistributed(MobileNet(input_shape=(108,192,3), alpha=0.25, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling='avg'))(input_A)
    
    image_model = Bidirectional(LSTM(16, activation=None))(image_model)
    
    image_model = BatchNormalization()(image_model)

    image_model = Activation('relu')(image_model)
    
    input_B = Input(shape=(128,259,1))
    
    audio_model = MobileNet(input_shape=(128,259,1), alpha=0.25, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling='avg')(input_B)
    
    audio_model = Dense(32)(audio_model)

    audio_model = BatchNormalization()(audio_model)

    audio_model = Activation('relu')(audio_model)
    
    model = Concatenate()([image_model, audio_model])
    
    model = Dense(1, activation='sigmoid')(model)
    
    model = Model(inputs=[input_A, input_B], outputs=model)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print_summary(model, line_length=120)
    #plot_model(model, show_shapes=True, to_file='model.png')
    
    return model

def get_2d_lstm_model_image_only():

    input_A = Input(shape=(None,108,192,3))
    
    image_model = TimeDistributed(MobileNet(input_shape=(108,192,3), alpha=0.25, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling='avg'))(input_A)
    
    image_model = Bidirectional(LSTM(16, activation=None))(image_model)
    
    image_model = BatchNormalization()(image_model)

    image_model = Activation('relu')(image_model)

    model = Dense(1, activation='sigmoid')(image_model)
    
    model = Model(inputs=input_A, outputs=model)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print_summary(model, line_length=120)
    #plot_model(model, show_shapes=True, to_file='model.png')
    
    return model



def get_3d_cnn_model_image_only():
    inputA = Input(shape=(4,108,192,3))
 
    model = Convolution3D(filters=32, kernel_size=(3,3,3),
                          activation='relu',
                          padding='same', data_format='channels_last')(inputA)

    model = MaxPooling3D(pool_size=(2,2,2))(model)

    model = Flatten()(model)

    model = Dense(1, activation = 'sigmoid')(model)

    model = Model(inputs=inputA, outputs=model)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print_summary(model, line_length=120)

    return model



def get_simple_cnn_model_image_only():

    input_A = Input(shape=(108,192,3))
    
    image_model = MobileNet(input_shape=(108,192,3), alpha=0.25, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling='avg')(input_A)

    model = Dense(1, activation='sigmoid')(image_model)
    
    model = Model(inputs=input_A, outputs=model)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print_summary(model, line_length=120)
    #plot_model(model, show_shapes=True, to_file='model.png')
    
    return model


def get_2d_lstm_model_8():

    input_A = Input(shape=(None,108,192,3))
    
    image_model = TimeDistributed(MobileNet(input_shape=(108,192,3), alpha=0.25, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling='avg'))(input_A)
    
    image_model = Bidirectional(LSTM(16, activation=None))(image_model)
    
    image_model = BatchNormalization()(image_model)

    image_model = Activation('relu')(image_model)
    
    input_B = Input(shape=(128,345,1))
    
    audio_model = MobileNet(input_shape=(128,345,1), alpha=0.25, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling='avg')(input_B)
    
    audio_model = Dense(32)(audio_model)

    audio_model = BatchNormalization()(audio_model)

    audio_model = Activation('relu')(audio_model)
    
    model = Concatenate()([image_model, audio_model])
    
    model = Dense(1, activation='sigmoid')(model)
    
    model = Model(inputs=[input_A, input_B], outputs=model)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print_summary(model, line_length=120)
    #plot_model(model, show_shapes=True, to_file='model.png')
    
    return model

def get_3d_cnn_model_image_audio():
    input_A = Input(shape=(4,108,192,3))
 
    image_model = Convolution3D(filters=32, kernel_size=(3,3,3),
                          activation='relu',
                          padding='same', data_format='channels_last')(input_A)

    image_model = MaxPooling3D(pool_size=(2,2,2))(image_model)

    image_model = Flatten()(image_model)

    image_model = Dense(32)(image_model)

    image_model = BatchNormalization()(image_model)

    image_model = Activation('relu')(image_model)

    input_B = Input(shape=(128,345,1))
    
    audio_model = MobileNet(input_shape=(128,345,1), alpha=0.25, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling='avg')(input_B)
    
    audio_model = Dense(32)(audio_model)

    audio_model = BatchNormalization()(audio_model)

    audio_model = Activation('relu')(audio_model)

    model = Concatenate()([image_model, audio_model])

    model = Dense(1, activation = 'sigmoid')(model)

    model = Model(inputs=[input_A, input_B], outputs=model)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print_summary(model, line_length=120)

    return model

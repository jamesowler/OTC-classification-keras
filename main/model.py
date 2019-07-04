from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import multi_gpu_model

from main.params import params


def CNN_plus_batch_norm():

    '''

    Classification of optical coherence tomography images

    :return:
    '''

    kernal_size = (3, 3)
    input_shape = (params['image_size_X'], params['image_size_Y'], 1)
    nb_pool = 2

    model = Sequential()

    # 32
    model.add(Conv2D(32, kernal_size, padding='same', activation='relu', input_shape=input_shape, name='Conv_0_0'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernal_size, padding='same', activation='relu', name='Conv_1_0'))
    model.add(BatchNormalization())
    # MP 2
    model.add(MaxPooling2D((nb_pool, nb_pool), padding='same'))

    # 64
    model.add((Conv2D(64, kernal_size, padding='same', activation='relu', name='Conv_2_1')))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernal_size, padding='same', activation='relu', name='Conv_3_1'))
    model.add(BatchNormalization())
    # MP 4
    model.add(MaxPooling2D((nb_pool, nb_pool), padding='same'))

    # 128
    model.add((Conv2D(64, kernal_size, padding='same', activation='relu', name='Conv_4_2')))
    model.add(BatchNormalization())
    model.add((Conv2D(128, kernal_size, padding='same', activation='relu', name='Conv_5_2')))
    model.add(BatchNormalization())
    # MP 8
    model.add(MaxPooling2D((nb_pool, nb_pool), padding='same'))

    # 256
    model.add((Conv2D(128, kernal_size, padding='same', activation='relu', name='Conv_6_3')))
    model.add(BatchNormalization())
    model.add((Conv2D(256, kernal_size, padding='same', activation='relu', name='Conv_7_3')))
    model.add(BatchNormalization())
    # MP 16
    model.add(MaxPooling2D((nb_pool, nb_pool), padding='same'))
    model.add((Conv2D(256, kernal_size, padding='same', activation='relu', name='Conv_8_3')))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(params['num_classes']))
    model.add(Activation('softmax'))

    if params['nb_GPUS'] < 1:
        model = multi_gpu_model(model, gpus=params['nb_GPUs'])

    model.compile(optimizer=Adam(lr=float(params['learning_rate'])), loss=params['loss_method'])

    return model

if __name__ == '__main__':

    model = CNN_plus_batch_norm()

    print(model.summary())

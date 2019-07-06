from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import multi_gpu_model
import numpy as np
from keras import backend as K

from params import params

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
    model.add(Conv2D(64, kernal_size, padding='same', activation='relu', name='Conv_0_1'))
    model.add(BatchNormalization())
    # MP 2
    model.add(MaxPooling2D((nb_pool, nb_pool), padding='same'))

    # 64
    model.add((Conv2D(64, kernal_size, padding='same', activation='relu', name='Conv_1_0')))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernal_size, padding='same', activation='relu', name='Conv_1_1'))
    model.add(BatchNormalization())
    # MP 4
    model.add(MaxPooling2D((nb_pool, nb_pool), padding='same'))

    # 128
    model.add((Conv2D(128, kernal_size, padding='same', activation='relu', name='Conv_2_0')))
    model.add(BatchNormalization())
    model.add((Conv2D(256, kernal_size, padding='same', activation='relu', name='Conv_2_1')))
    model.add(BatchNormalization())
    # MP 8
    model.add(MaxPooling2D((nb_pool, nb_pool), padding='same'))

    # 256
    model.add((Conv2D(256, kernal_size, padding='same', activation='relu', name='Conv_3_0')))
    model.add(BatchNormalization())
    model.add((Conv2D(512, kernal_size, padding='same', activation='relu', name='Conv_3_1')))
    model.add(BatchNormalization())
    # MP 16
    model.add(MaxPooling2D((nb_pool, nb_pool), padding='same'))

    # 512
    model.add((Conv2D(512, kernal_size, padding='same', activation='relu', name='Conv_4_0')))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(params['num_classes']))
    model.add(Activation('softmax'))

    if params['nb_GPUs'] < 1:
        model = multi_gpu_model(model, gpus=params['nb_GPUs'])

    model.compile(optimizer=Adam(lr=float(params['learning_rate'])), loss=params['loss_method'])

    return model


def get_model_memory_usage(batch_size, model):
    '''
    Calculates model memory requirements
    '''

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)

    return gbytes


if __name__ == '__main__':

    model = CNN_plus_batch_norm()

    gbytes = get_model_memory_usage(params['batch_size'], model)
    print(f'\n Model requires: {gbytes} GB of memory \n')
    print(model.summary())

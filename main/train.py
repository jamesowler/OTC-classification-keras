import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from main.params import params
from main.model import CNN_plus_batch_norm
import glob

model = CNN_plus_batch_norm()


def data_generator(dir_path):

    datagen = ImageDataGenerator(zoom_range=0.1,
                                   rotation_range=3,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   horizontal_flip=True,
                                   fill_mode='constant',
                                   cval=0)

    generator = datagen.flow_from_directory(dir_path, batch_size=params['batch_size'], target_size=(params['image_size_X'], params['image_size_Y']), color_mode='grayscale')

    steps_per_epoch = np.ceil(len(glob.glob(dir_path + '/*/*'))/params['batch_size'])

    return generator, steps_per_epoch


def train():

    '''
    Runs training of the model with a progress bar
    :return:
    '''

    train_generator, train_steps_per_epoch = data_generator('../data/OCT2017/train')
    val_generator, val_steps_per_epoch = data_generator('../data/OCT2017/val')
    model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=params['nb_epoch'],
                        validation_data=val_generator, validation_steps=val_steps_per_epoch, verbose=1)


if __name__ == '__main__':

    print('Begin model training \n')

    train()
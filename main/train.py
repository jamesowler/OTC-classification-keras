import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from .params import params
from .model import CNN_plus_batch_norm
import glob
import os

model = CNN_plus_batch_norm()


def data_generator(dir_path):

    '''
    Creates generator that can perform image augmentation on-the-fly
    '''

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
    Runs training of the model with progress bar
    :return:
    '''

    train_generator, train_steps_per_epoch = data_generator('../data/OCT2017/train')
    val_generator, val_steps_per_epoch = data_generator('../data/OCT2017/val')

    if not os.path.exists('../saved_models'):
        os.mkdir('../saved_models')

    # saved new model (overwrites current model) every 5 epochs if validation loss is less than previous model
    checkpoint_save = ModelCheckpoint(f'../saved_models/{params["model_name"]}' + '{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=False, mode='auto', period=5)

    checkpoint_list = [checkpoint_save]
    history = model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=params['nb_epoch'],
                        validation_data=val_generator, validation_steps=val_steps_per_epoch, verbose=1, checkpoint=checkpoint_list)

    return history


if __name__ == '__main__':

    from .validation import plot_losses

    print('Begin model training \n')

    history = train()
    plot_losses(history)
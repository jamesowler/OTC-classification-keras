import matplotlib.pyplot as plt
import numpy as np

def plot_losses(history_object, filepath='../train_vs_val_loss.png'):

    '''
    :param history_object:
    :return:
    '''

    plt.plot(history_object['loss'])
    plt.plot(history_object['val_loss'])
    plt.title('Training vs validation accuracy')
    plt.savefig(filepath)
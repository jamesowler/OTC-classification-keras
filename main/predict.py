import glob
import argparse
from keras.models import load_model
import data_utils
from params import params
import numpy as np


def inference_from_model_file(file_path, model_file_path='./model.h5'):

    X_final = np.zeros(1, params['image_size_X'])
    X = main.data_utils.preprocess(file_path, params['image_size_X'], params['image_size_Y'])
    model = load_model(model_file_path)
    Y_pred = model.predict(X)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='/path/to/file/for/prediction')
    args = parser.parse_args()

    file_path = args.input
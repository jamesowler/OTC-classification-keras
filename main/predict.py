import glob
from keras.models import load_model
import main.data_utils

def inference_from_model_file(X, model_file_path='./model.h5'):

    model = load_model(model_file_path)
    Y_pred = model.predict(X)

import os
import numpy as np
import cv2
import glob



def find_median_dimensions():
    files = glob.glob('/Users/jamesowler/Projects/Deep-learning/OTC/data/OCT2017/test/*/*')

    print(files)

    shapes = []

    for i in files:
        img  = cv2.imread(i, 0)
        shapes.append(img.shape)

        if img.shape[1] > 780:
            print(i)

    print(set(shapes))

    print(shapes)


def preprocess(input_image, xshape, yshape):

    '''
    :param input_image: gray-scale image numpy array
    :returns: resized padded array
    '''

    # normalise intensities
    high = 255.0
    low = 0.0
    min = np.min(input_image)
    max = np.max(input_image)
    range_ = max - min
    img_processed = high - (((high - low) * (max - input_image)) / range_)

    # resize
    cv2.resize(img_processed, (xshape, yshape), interpolation=cv2.INTER_LINEAR)

    return img_processed


def load_cases(fnames, xshape, yshape, num_classes):

    '''
    :param fnames: Full file paths of image file for that batch
    :param xshape: Image dimension in x axis
    :param yshape: Image dimension in y axis
    :param num_classes: Number of classification classes
    :return:
    '''

    X = np.zeros((len(fnames), xshape, yshape, 1), dtype='float32')
    Y = np.zeros((len(fnames), num_classes, 1), dtype='float32')

    for n, i in enumerate(fnames):
        # one hot encoding
        class_name = os.path.basename(os.path.dirname(i))
        class_index = classes.index(class_name)
        class_label = np.zeros((num_classes))
        class_label[class_index] = 1

        img = cv2.imread(i, 0)
        img = np.rot90(preprocess(img, xshape, yshape))
        print(img.shape)

        X[n, :, :, 0] = img[:, :]
        Y[n, :, 0] = class_label

    return X, Y


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # test image loader works properly
    X, Y = load_cases(['/Users/jamesowler/Projects/Deep-learning/OTC/data/OCT2017/test/CNV/CNV-53018-2.jpeg'], 512, 496, 4)
    plt.imshow(X[0, :, :, 0], cmap='gray')
    plt.show()


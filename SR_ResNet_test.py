from SR_ResNet import EDSRModel, make_model, PSNR, plot_results
import numpy as np
import tensorflow as tf
from data_loader import data_loader
import matplotlib.pyplot as plt
from model import FacialRecognitionModel
from tensorflow import keras
from keras import layers

AUTOTUNE = tf.data.AUTOTUNE
from keras.utils import to_categorical
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_lfw_pairs
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model



if __name__ == "__main__":

    data_set = data_loader(color=True)
    X_train, y_train = data_set.get_train()
    X_val, y_val = data_set.get_val()
    X_test, y_test = data_set.get_test()
    # convert images to integars from floats
    X_train = (X_train * 255).astype(np.uint8)
    X_val = (X_val * 255).astype(np.uint8)
    X_test = (X_test * 255).astype(np.uint8)
    # create variable for low res and high res data to train model
    X_train_lr = resize(X_train, (X_train.shape[0], 16, 11, 3))
    X_train_hr = resize(X_train, (X_train.shape[0], 144, 99, 3), anti_aliasing=True, order=0)

    X_val_lr = resize(X_val, (X_val.shape[0], 16, 11, 3))
    X_val_hr = resize(X_val, (X_val.shape[0], 144, 99, 3), anti_aliasing=True, order=0)

    X_test_lr = resize(X_test, (X_test.shape[0], 16, 11, 3))
    X_test_hr = resize(X_test, (X_test.shape[0], 144, 99, 3), anti_aliasing=True, order=0)
    # Build and Train FR model on HR data split
    fr_model = FacialRecognitionModel()
    fr_model.load()

    model = make_model(num_filters=64, num_of_residual_blocks=16)

    # Using adam optimizer with initial learning rate as 1e-4, changing learning rate after 5000 steps to 5e-5
    optim_edsr = keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[5000], values=[1e-4, 5e-5]
        )
    )
    # Compiling model with loss as mean absolute error(L1 Loss) and metric as psnr
    model.compile(optimizer=optim_edsr, loss="mae", metrics=[PSNR])

    model = load_model("./models/SRResNet.keras")

    # get various LR image sizes
    X_test_8 = resize(X_test, (X_test.shape[0], 16, 11, 3))
    X_test_6 = resize(X_test, (X_test.shape[0], 21, 16, 3))
    X_test_4 = resize(X_test, (X_test.shape[0], 31, 24, 3))
    X_test_2 = resize(X_test, (X_test.shape[0], 63, 47, 3))

    # make predictions for each LR size
    predsby8 = np.zeros((X_test.shape[0], 144, 99, 3), dtype=np.uint8)
    predsby6 = np.zeros((X_test.shape[0], 189, 144, 3), dtype=np.uint8)
    predsby4 = np.zeros((X_test.shape[0], 279, 216, 3), dtype=np.uint8)
    predsby2 = np.zeros((X_test.shape[0], 567, 423, 3), dtype=np.uint8)
    # add prediction to empty lists declared above
    for im in range(len(X_test)):
        predsby8[im] = model.predict_step(X_test_8[im])
        predsby6[im] = model.predict_step(X_test_6[im])
        predsby4[im] = model.predict_step(X_test_4[im])
        predsby2[im] = model.predict_step(X_test_2[im])
    # print shapes
    '''
    print(predsby8.shape)
    print(predsby6.shape)
    print(predsby4.shape)
    print(predsby2.shape)
    '''
    # resize prediction images for FR model
    predsby8 = resize(predsby8, (predsby8.shape[0], 144, 99, 3), anti_aliasing=True, order=0)
    predsby6 = resize(predsby6, (predsby6.shape[0], 144, 99, 3), anti_aliasing=True, order=0)
    predsby4 = resize(predsby4, (predsby4.shape[0], 144, 99, 3), anti_aliasing=True, order=0)
    predsby2 = resize(predsby2, (predsby2.shape[0], 144, 99, 3), anti_aliasing=True, order=0)
    # test FR model on SR images
    fr_model.test(predsby8, to_categorical(y_test))
    fr_model.test(predsby6, to_categorical(y_test))
    fr_model.test(predsby4, to_categorical(y_test))
    fr_model.test(predsby2, to_categorical(y_test))
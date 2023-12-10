from SR_ResNet import EDSRModel, make_model, PSNR, plot_results
import numpy as np
import tensorflow as tf
from data_loader import data_loader
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers

AUTOTUNE = tf.data.AUTOTUNE

from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_lfw_pairs
from skimage.transform import resize
from sklearn.model_selection import train_test_split


if __name__ =="__main__":
    data_set = data_loader(color=True)  # declare variable for dataset
    # get training, validation, and testing data
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
    # print data shapes to check
    print(X_train_lr.shape)
    print(X_train_hr.shape)

    model = make_model(num_filters=64, num_of_residual_blocks=16)

    # Using adam optimizer with initial learning rate as 1e-4, changing learning rate after 5000 steps to 5e-5
    optim_edsr = keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[5000], values=[1e-4, 5e-5]
        )
    )
    # Compiling model with loss as mean absolute error(L1 Loss) and metric as psnr
    model.compile(optimizer=optim_edsr, loss="mae", metrics=[PSNR])
    # Training for more epochs will improve results
    history = model.fit(X_train_lr, X_train_hr, epochs=100, steps_per_epoch=200, validation_data=(X_val_lr, X_val_hr))
    model.save('./models/SRResNet.keras')

    preds = model.predict_step(X_test_lr[0])
    plot_results(X_test_hr[0], X_test_lr[0], preds)

    # plot metrics from training
    # loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('SR_loss-plt.png')
    # PSNR plot
    plt.figure()
    plt.plot(history.history['PSNR'], label='PSNR')
    plt.plot(history.history['val_PSNR'], label='val_PSNR')
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch #')
    plt.ylabel('PSNR')
    plt.legend()

    plt.savefig('./figures/SR_PSNR-plt.png')
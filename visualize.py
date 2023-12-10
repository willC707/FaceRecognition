from sr_model import SRGAN
from model import FacialRecognitionModel
from data_loader import data_loader
from skimage.transform import resize
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

if __name__ == "__main__":

    data = data_loader(color=True)

    model = SRGAN(lr=(63,47,3), hr=(252, 188, 3))
    model.load(res='best3')

    X_test, y_test = data.get_test()

    X_test_lr = data.get_resized(split="test",size=[63,47,3])

    X_test_sr = model.gen_sr_split(X_test_lr)

    fig = plt.figure()
    ax = fig.add_subplot(1,3,1)
    ax.imshow(X_test[0])

    ax = fig.add_subplot(1,3,2)
    ax.imshow(X_test_lr[0])

    ax = fig.add_subplot(1,3,3)
    ax.imshow(X_test_sr[0])

    plt.savefig(f"./figures/155ep_lr2.png")
    plt.clf()

    X_test_lr = data.get_resized(split="test", size=[31, 24, 3])
    X_test_lr = resize(X_test_lr, (X_test_lr.shape[0], 63,47,3), anti_aliasing=True, order=0)

    X_test_sr = model.gen_sr_split(X_test_lr)

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(X_test[0])

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(X_test_lr[0])

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(X_test_sr[0])

    plt.savefig(f"./figures/155ep_lr4.png")
    plt.clf()

    X_test_lr = data.get_resized(split="test", size=[21, 16, 3])
    X_test_lr = resize(X_test_lr, (X_test_lr.shape[0], 63, 47, 3), anti_aliasing=True, order=0)

    X_test_sr = model.gen_sr_split(X_test_lr)

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(X_test[0])

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(X_test_lr[0])

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(X_test_sr[0])

    plt.savefig(f"./figures/155ep_lr6.png")
    plt.clf()

    X_test_lr = data.get_resized(split="test", size=[16, 11, 3])
    X_test_lr = resize(X_test_lr, (X_test_lr.shape[0], 63, 47, 3), anti_aliasing=True, order=0)

    X_test_sr = model.gen_sr_split(X_test_lr)

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(X_test[0])

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(X_test_lr[0])

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(X_test_sr[0])

    plt.savefig(f"./figures/155ep_lr8.png")
    plt.clf()
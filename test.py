from sr_model import SRGAN
from model import FacialRecognitionModel
from data_loader import data_loader
from skimage.transform import resize
import numpy as np
from keras.utils import to_categorical


if __name__=="__main__":
    data = data_loader(color=True)
    test_hr, y_test = data.get_test()

    fr_model = FacialRecognitionModel()
    fr_model.load()
    srmodel = SRGAN(lr=(63,47,3), hr=(252, 188, 3))
    srmodel.load(res="best3")

    print("HR Baseline:")
    fr_model.test(test_hr, to_categorical(y_test))

    test_sr = data.get_resized(split="test",size=[63,47,3])
    test_lr = resize(test_sr, (test_sr.shape[0],125,94,3), anti_aliasing=True, order=0)
    print("LR /2 Baseline:")
    fr_model.test(test_lr, to_categorical(y_test))
    print("SR /2 Test:")
    test_lr = srmodel.gen_sr_split(test_sr)
    test_lr = np.array(test_lr)
    test_lr = resize(test_lr, (test_lr.shape[0],125,94,3), anti_aliasing=True, order=0)
    fr_model.test(test_lr, to_categorical(y_test))

    test_sr = data.get_resized(split="test", size=[31, 24, 3])
    test_lr = resize(test_sr, (test_sr.shape[0], 125, 94, 3), anti_aliasing=True, order=0)
    test_sr = resize(test_sr, (test_sr.shape[0], 64, 47, 3), anti_aliasing=True, order=0)
    print("LR /4 Baseline:")
    fr_model.test(test_lr, to_categorical(y_test))
    print("SR /4 Test:")
    test_lr = srmodel.gen_sr_split(test_sr)
    test_lr = np.array(test_lr)
    test_lr = resize(test_lr, (test_lr.shape[0], 125, 94, 3), anti_aliasing=True, order=0)
    fr_model.test(test_lr, to_categorical(y_test))

    test_sr = data.get_resized(split="test", size=[21, 16, 3])
    test_lr = resize(test_sr, (test_sr.shape[0], 125, 94, 3), anti_aliasing=True, order=0)
    test_sr = resize(test_sr, (test_sr.shape[0], 64, 47, 3), anti_aliasing=True, order=0)
    print("LR /6 Baseline:")
    fr_model.test(test_lr, to_categorical(y_test))
    print("SR /6 Test:")
    test_lr = srmodel.gen_sr_split(test_sr)
    test_lr = np.array(test_lr)
    test_lr = resize(test_lr, (test_lr.shape[0], 125, 94, 3), anti_aliasing=True, order=0)
    fr_model.test(test_lr, to_categorical(y_test))

    test_sr = data.get_resized(split="test", size=[16, 11, 3])
    test_lr = resize(test_sr, (test_sr.shape[0], 125, 94, 3), anti_aliasing=True, order=0)
    test_sr = resize(test_sr, (test_sr.shape[0], 64, 47, 3), anti_aliasing=True, order=0)
    print("LR /8 Baseline:")
    fr_model.test(test_lr, to_categorical(y_test))
    print("SR /8 Test:")
    test_lr = srmodel.gen_sr_split(test_sr)
    test_lr = np.array(test_lr)
    test_lr = resize(test_lr, (test_lr.shape[0], 125, 94, 3), anti_aliasing=True, order=0)
    fr_model.test(test_lr, to_categorical(y_test))

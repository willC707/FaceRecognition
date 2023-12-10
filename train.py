from data_loader import data_loader
from model import FacialRecognitionModel
from sr_model import SRGAN
from keras.utils import to_categorical
from skimage.transform import resize


if __name__ == "__main__":

    dataloader = data_loader(color=True)

    X_train_hr, y_train = dataloader.get_train()
    X_val_hr, y_val = dataloader.get_val()

    model = FacialRecognitionModel()
    model.train(X_train_hr.reshape(-1, 125, 94, 3), to_categorical(y_train),
                X_val_hr.reshape(-1, 125, 94, 3), to_categorical(y_val))

    model.save()
    X_train_lr = resize(X_train_hr, (X_train_hr.shape[0], 16, 11,3))
    X_train_lr = resize(X_train_lr, (X_train_lr.shape[0], 63, 47, 3), anti_aliasing=True, order=0)
    X_train_hr = resize(X_train_hr, (X_train_hr.shape[0], 252, 188, 3), anti_aliasing=True, order=0)
    model = SRGAN(lr=(63,47,3), hr=(252, 188, 3))
    model.train(X_train_lr, X_train_hr)
    model.save(res="final")
    # X_test_hr, y_test = dataloader.get_test()
    # print("High Resolution Testing")
    # model.test(X_test_hr.reshape(-1, 125, 94, 3), to_categorical(y_test))
    #
    # X_test_lr = dataloader.get_resized(split="test", size=[63,47,3])
    # X_test_lr = resize(X_test_lr,(X_test_lr.shape[0], 125, 94, 3), anti_aliasing=True, order=0)
    #
    # print("Low Resolution Testing: /2")
    # model.test(X_test_lr.reshape(-1,125,94,3), to_categorical(y_test))
    #
    # X_test_lr = dataloader.get_resized(split="test", size=[31, 24, 3])
    # X_test_lr = resize(X_test_lr, (X_test_lr.shape[0], 125, 94, 3), anti_aliasing=True, order=0)
    #
    # print("Low Resolution Testing: /4")
    # model.test(X_test_lr.reshape(-1, 125, 94, 3), to_categorical(y_test))
    #
    # X_test_lr = dataloader.get_resized(split="test", size=[21, 16, 3])
    # X_test_lr = resize(X_test_lr, (X_test_lr.shape[0], 125, 94, 3), anti_aliasing=True, order=0)
    #
    # print("Low Resolution Testing: /6")
    # model.test(X_test_lr.reshape(-1, 125, 94, 3), to_categorical(y_test))
    #
    # X_test_lr = dataloader.get_resized(split="test", size=[16, 11, 3])
    # X_test_lr = resize(X_test_lr, (X_test_lr.shape[0], 125, 94, 3), anti_aliasing=True, order=0)
    #
    # print("Low Resolution Testing: /8")
    # model.test(X_test_lr.reshape(-1, 125, 94, 3), to_categorical(y_test))


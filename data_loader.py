
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_lfw_pairs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from model import FacialRecognitionModel
from keras.utils import to_categorical
from sklearn.preprocessing import normalize
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt


class data_loader():
    def __init__(self,color):
        self.lfw_people = fetch_lfw_people(data_home="./data", color=color, resize=None, min_faces_per_person=70,
                                           funneled=False,)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.lfw_people.images,
                                                                                self.lfw_people.target,
                                                                                test_size=0.25,
                                                                                random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                              test_size=0.10, random_state=42)

    def get_train(self):
        return self.X_train, self.y_train

    def get_val(self):
        return self.X_val, self.y_val

    def get_test(self):
        return self.X_test, self.y_test

    def get_resized(self, split="train", size=[25,25,3]):
        if split=="train":
            X_resized = resize(self.X_train, (self.X_train.shape[0], size[0], size[1], size[2]))
        if split=="test":
            X_resized = resize(self.X_test, (self.X_test.shape[0], size[0], size[1], size[2]))
        if split=="val":
            X_resized = resize(self.X_val, (self.X_val.shape[0], size[0], size[1], size[2]))
        return X_resized




if __name__ == "__main__":

    print('Loading Data')
    # lfw_people = fetch_lfw_pairs(data_home="./data", subset='train', color=True, resize=None)
    lfw_people = fetch_lfw_people(data_home="./data", color=True, resize=None, min_faces_per_person=70, funneled=False)
    # lfw_pairs = fetch_lfw_pairs(data_home="./data",color=True, resize=None,funneled=False,slice_=None)
    # lfw_people.images = normalize(lfw_people.images.reshape((lfw_people.images.shape[0], -1)))
    print(lfw_people.images.shape)
    print(lfw_people.target.shape)

    print('Splitting Data')
    X_train, X_test, y_train, y_test = train_test_split(lfw_people.images,lfw_people.target, test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
    X_test_lr = resize(X_test, (X_test.shape[0], 13, 9, 3))
    plt.imshow(X_test_lr[0])
    plt.show()
    X_test_lr = resize(X_test_lr, (X_test_lr.shape[0], 125, 94, 3), anti_aliasing=True, order=0)
    plt.imshow(X_test_lr[0])
    plt.show()
    plt.imshow(X_test[0])
    plt.show()

    print(X_train.shape)
    print(y_train.shape)

    # X_train = X_train.astype('float32') / 255
    # X_test = X_test.astype('float32') / 255
    # X_train = np.reshape(X_train, (X_train.shape[0], 250, 250, 3))
    # X_test = np.reshape(X_test, (X_test.shape[0], 250, 250, 3))

    model = FacialRecognitionModel()
    model.train(X_train.reshape(-1, 125,94,3), to_categorical(y_train), X_val.reshape(-1, 125,94,3), to_categorical(y_val))

    model.test(X_test.reshape(-1, 125,94,3), to_categorical(y_test))


    model.test(X_test_lr.reshape(-1, 125, 94,3), to_categorical(y_test))

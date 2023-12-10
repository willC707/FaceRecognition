from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import keras


class FacialRecognitionModel:
    def __init__(self, in_shape=(125,94,3)):
        self._model = Sequential()

        self._model.add(Conv2D(32, (5,5), activation='relu', input_shape=in_shape))
        self._model.add(MaxPooling2D(pool_size=(2,2)))
        self._model.add(Conv2D(64, (3,3,) , activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2,2)))
        self._model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2,2)))
        self._model.add(Flatten())
        self._model.add(Dense(units=256, activation='relu'))
        self._model.add(Dense(units=7, activation='softmax'))

        self._model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val):
        self._model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))

    def test(self, X_test, y_test):
        score = self._model.evaluate(X_test, y_test, verbose=0)
        print(f'Test loss: {score[0]:.4f}')
        print(f'Test accuracy: {score[1]:.4f}')

    def save(self):
        self._model.save("./models/fr_model.h5")

    def load(self):
        self._model = keras.models.load_model("./models/fr_model.h5")
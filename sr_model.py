import numpy as np
from keras import Model
from keras.layers import Conv2D, PReLU,BatchNormalization, Flatten, Add, MaxPooling2D, Activation
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add, ReLU, Reshape, Dropout
from keras.applications import VGG19
from keras import models
from keras.applications import VGG19
from tqdm.auto import tqdm
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt


class SRGAN():
    def __init__(self,lr=(16, 11, 3), hr=(144, 99, 3)):
        self.lr_ip = Input(shape=lr)
        self.hr_ip = Input(shape=hr)
        #self.train_lr,self.train_hr = train_lr, train_hr#training images arrays normalized between 0 & 1
        #self.test_lr, self.test_hr = test_lr, test_hr# testing images arrays normalized between 0 & 1
        self.generator = self.create_gen(self.lr_ip)
        self.discriminator = self.create_disc(self.hr_ip)
        self.discriminator.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001),
                              metrics=['accuracy'])
        self.vgg = self.build_vgg(hr=hr)
        self.vgg.trainable = False
        self.gan_model = self.create_comb(self.generator, self.discriminator, self.vgg, self.lr_ip, self.hr_ip)
        self.gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=
        [1e-3, 1], optimizer=Adam(learning_rate=0.001))

# Residual block
# Residual block
    def res_block(self,ip):

        res_model = Conv2D(64, (3,3), padding = "same")(ip)
        res_model = BatchNormalization(momentum = 0.5)(res_model)
        res_model = PReLU(shared_axes = [1,2])(res_model)

        res_model = Conv2D(64, (3,3), padding = "same")(res_model)
        res_model = BatchNormalization(momentum = 0.5)(res_model)

        return add([ip,res_model])

# Upscale the image 2x
    def upscale_block(self,ip):
        up_model = Conv2D(256, (3,3), padding="same")(ip)
        up_model = UpSampling2D( size = 2 )(up_model)
        up_model = PReLU(shared_axes=[1,2])(up_model)

        return up_model


# Generator Model
    def create_gen(self,gen_ip):
        num_res_block = 16
        layers = Conv2D(64, (9,9), padding="same")(gen_ip)
        layers = PReLU(shared_axes=[1,2])(layers)
        temp = layers
        for i in range(num_res_block):
            layers = self.res_block(layers)
        layers = Conv2D(64, (3,3), padding="same")(layers)
        layers = BatchNormalization(momentum=0.5)(layers)
        layers = add([layers,temp])
        layers = self.upscale_block(layers)
        layers = self.upscale_block(layers)
        op = Conv2D(3, (9,9), padding="same")(layers)
        return Model(inputs=gen_ip, outputs=op)

#Small block inside the discriminator
    def discriminator_block(self, ip, filters, strides=1, bn=True):

        disc_model = Conv2D(filters, (3,3), strides, padding="same")(ip)
        disc_model = LeakyReLU( alpha=0.2 )(disc_model)
        if bn:
            disc_model = BatchNormalization( momentum=0.8 )(disc_model)
        return disc_model

# Discriminator Model
    def create_disc(self,disc_ip):
        df = 64

        d1 = self.discriminator_block(disc_ip, df, bn=False)
        d2 = self.discriminator_block(d1, df, strides=2)
        d3 = self.discriminator_block(d2, df*2)
        d4 = self.discriminator_block(d3, df*2, strides=2)
        d5 = self.discriminator_block(d4, df*4)
        d6 = self.discriminator_block(d5, df*4, strides=2)
        d7 = self.discriminator_block(d6, df*8)
        d8 = self.discriminator_block(d7, df*8, strides=2)

        d8_5 = Flatten()(d8)
        d9 = Dense(df*16)(d8_5)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)
        return Model(disc_ip, validity)



# Build the VGG19 model upto 10th layer
# Used to extract the features of high res imgaes
    def build_vgg(self, hr):
        #img = Input(shape=(224, 224, 3))
        vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr)
        output = vgg.outputs = [vgg.layers[9].output]

        #img_features = vgg(img)
        return Model(inputs=vgg.input, outputs=output)


# Attach the generator and discriminator
    def create_comb(self,gen_model, disc_model, vgg, lr_ip, hr_ip):
        gen_img = gen_model(lr_ip)
        gen_features = vgg(gen_img)
        disc_model.trainable = False
        validity = disc_model(gen_img)
        return Model([lr_ip, hr_ip],[validity,gen_features])

    def train(self, train_lr, train_hr):
        batch_size = 1
        train_lr_batches = []
        train_hr_batches = []
        for it in range(int(train_hr.shape[0] / batch_size)):
            start_idx = it * batch_size
            end_idx = start_idx + batch_size
            train_hr_batches.append(train_hr[start_idx:end_idx])
            train_lr_batches.append(train_lr[start_idx:end_idx])
        train_lr_batches = np.array(train_lr_batches)
        train_hr_batches = np.array(train_hr_batches)

        epochs = 30
        for e in range(epochs):
            # print(f'{e}')
            gen_label = np.zeros((batch_size, 1))
            real_label = np.ones((batch_size, 1))
            g_losses = []
            d_losses = []
            for b in tqdm(range(len(train_hr_batches))):
                # print(f'{b}')
                # lr_imgs = train_lr_batches[b]
                # hr_imgs = train_hr_batches[b]
                gen_imgs = self.generator.predict_on_batch(train_lr_batches[b])
                # Dont forget to make the discriminator trainable
                self.discriminator.trainable = True

                # Train the discriminator
                d_loss_gen = self.discriminator.train_on_batch(gen_imgs, gen_label)
                d_loss_real = self.discriminator.train_on_batch(train_hr_batches[b],
                                                                real_label)
                self.discriminator.trainable = False
                d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
                image_features = self.vgg.predict(train_hr_batches[b], verbose=0)
                # Train the generator
                g_loss, _, _ = self.gan_model.train_on_batch([train_lr_batches[b], train_hr_batches[b]],
                                                             [real_label, image_features], )
                d_losses.append(d_loss)
                g_losses.append(g_loss)
            g_losses = np.array(g_losses)
            d_losses = np.array(d_losses)
            g_loss = np.sum(g_losses, axis=0) / len(g_losses)
            d_loss = np.sum(d_losses, axis=0) / len(d_losses)
            print("epoch:", e + 1, "g_loss:", g_loss, "d_loss:", d_loss)

    def test(self, X_test_lr, X_test_hr):
        score = self.gan_model.evaluate(X_test_lr, X_test_hr)
        print(f'Test_data: {score}')

    def predict(self, X_lr, X_hr):
        # plt.imshow(X_hr)
        # plt.show()
        batch_size = 1
        test_lr_batches = []
        test_hr_batches = []
        for it in range(int(X_lr.shape[0] / batch_size)):
            start_idx = it * batch_size
            end_idx = start_idx + batch_size
            test_hr_batches.append(X_hr[start_idx:end_idx])
            test_lr_batches.append(X_lr[start_idx:end_idx])
        test_lr_batches = np.array(test_lr_batches)
        test_hr_batches = np.array(test_hr_batches)
        prediction = self.generator.predict_on_batch(test_lr_batches[0])
        print(prediction)
        print(type(prediction))
        plt.imshow(prediction[0])
        plt.show()

    def gen_sr_split(self, X_lr):
        batch_size = 23
        lr_batches = []
        hr_split = []
        for it in range(int(X_lr.shape[0] / batch_size)):
            start_idx = it * batch_size
            end_idx = start_idx + batch_size

            lr_batches.append(X_lr[start_idx:end_idx])
        lr_batches = np.array(lr_batches)
        for b in range(len(lr_batches)):
            batch_prediction = self.generator.predict_on_batch(lr_batches[b])
            for im in range(len(batch_prediction)):
                hr_split.append(batch_prediction[im])

        return hr_split

    def save(self, res):
        self.generator.save(f"/.models/sr_generator_{res}.h5")
        self.discriminator.save(f"/.models/sr_discriminator_{res}.h5")
        self.gan_model.save(f"./models/sr_gan_model_{res}.h5")
        self.vgg.save(f"./models/vgg_{res}.h5")

    def load(self, res):
        self.generator = keras.models.load_model(f"./models/sr_generator_{res}.h5")
        self.discriminator = keras.models.load_model(
            f"./models/sr_discriminator_{res}.h5")
        self.gan_model = keras.models.load_model(f"./models/sr_gan_model_{res}.h5")
        self.vgg = keras.models.load_model(f"./models/vgg_{res}.h5")
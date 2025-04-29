import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten, BatchNormalization,
    Conv2D, Conv2DTranspose, LeakyReLU, Reshape
)
from tensorflow.keras.models import Sequential
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
import time
from IPython import display

class DCGAN:
    def __init__(self, latent_dim=100, image_size=64):
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def _build_discriminator(self):
        model = Sequential([
            Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(self.image_size, self.image_size, 3)),
            LeakyReLU(alpha=0.2),
            Dropout(0.3),

            Conv2D(128, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dropout(0.3),

            Conv2D(256, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dropout(0.3),

            Conv2D(512, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dropout(0.3),

            Flatten(),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def _build_generator(self):
        model = Sequential([
            Dense(4 * 4 * 512, input_dim=self.latent_dim),
            Reshape((4, 4, 512)),

            Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'),
            BatchNormalization(momentum=0.8),
            LeakyReLU(alpha=0.2),

            Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
            BatchNormalization(momentum=0.8),
            LeakyReLU(alpha=0.2),

            Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
            BatchNormalization(momentum=0.8),
            LeakyReLU(alpha=0.2),

            Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')
        ])
        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def generate_images(self, num_examples=16):
        noise = tf.random.normal([num_examples, self.latent_dim])
        generated_images = self.generator(noise, training=False)
        generated_images = (generated_images + 1) * 127.5
        return generated_images

    def plot_generated_images(self, epoch=None, examples=16, dim=(4, 4)):
        generated_images = self.generate_images(examples)
        plt.figure(figsize=(10, 10))
        
        for i in range(examples):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_images[i].numpy().astype('uint8'))
            plt.axis('off')
        
        if epoch is not None:
            plt.suptitle(f'Epoch {epoch}')
        plt.show()
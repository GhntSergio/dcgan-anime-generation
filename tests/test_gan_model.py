import unittest
import tensorflow as tf
from gan_model import DCGAN

class TestDCGAN(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 100
        self.image_size = 64
        self.gan = DCGAN(latent_dim=self.latent_dim, image_size=self.image_size)

    def test_generator_output_shape(self):
        batch_size = 1
        noise = tf.random.normal([batch_size, self.latent_dim])
        generated_images = self.gan.generator(noise, training=False)
        
        self.assertEqual(generated_images.shape, 
                         (batch_size, self.image_size, self.image_size, 3))

    def test_discriminator_output_shape(self):
        batch_size = 1
        fake_images = tf.random.normal([batch_size, self.image_size, self.image_size, 3])
        output = self.gan.discriminator(fake_images, training=False)
        
        self.assertEqual(output.shape, (batch_size, 1))

    def test_generator_compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.gan.generator.compile(optimizer=optimizer)
        self.assertIsNotNone(self.gan.generator.optimizer)

    def test_discriminator_compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.gan.discriminator.compile(optimizer=optimizer)
        self.assertIsNotNone(self.gan.discriminator.optimizer)

if __name__ == '__main__':
    unittest.main()
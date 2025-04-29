import tensorflow as tf
from gan_model import DCGAN
import os
import time
from IPython import display
import pathlib

def prepare_dataset(data_dir, batch_size=32, image_size=64):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        label_mode=None,
        batch_size=batch_size,
        image_size=(image_size, image_size)
    )
    # Normalisation des images entre [-1, 1]
    dataset = dataset.map(lambda x: (x - 127.5) / 127.5)
    return dataset

@tf.function
def train_step(gan, images, batch_size, latent_dim):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gan.generator(noise, training=True)

        real_output = gan.discriminator(images, training=True)
        fake_output = gan.discriminator(generated_images, training=True)

        gen_loss = gan.generator_loss(fake_output)
        disc_loss = gan.discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(gen_loss, gan.generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, gan.discriminator.trainable_variables)

    gan.generator.optimizer.apply_gradients(
        zip(gen_gradients, gan.generator.trainable_variables)
    )
    gan.discriminator.optimizer.apply_gradients(
        zip(disc_gradients, gan.discriminator.trainable_variables)
    )

    return gen_loss, disc_loss

def train(gan, dataset, epochs=50, batch_size=32, latent_dim=100, save_interval=10):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator=gan.generator,
        discriminator=gan.discriminator
    )

    for epoch in range(epochs):
        start = time.time()
        print(f'\nÉpoque {epoch + 1}/{epochs}')

        for batch_idx, image_batch in enumerate(dataset):
            gen_loss, disc_loss = train_step(gan, image_batch, batch_size, latent_dim)
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, '
                      f'Perte G: {gen_loss:.4f}, '
                      f'Perte D: {disc_loss:.4f}')

        # Générer et sauvegarder les images
        display.clear_output(wait=True)
        gan.plot_generated_images(epoch + 1)

        # Sauvegarder le modèle périodiquement
        if (epoch + 1) % save_interval == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Temps pour époque {epoch + 1}: {time.time()-start:.2f} sec')

if __name__ == '__main__':
    # Paramètres d'entraînement
    BATCH_SIZE = 32
    EPOCHS = 100
    LATENT_DIM = 100
    IMAGE_SIZE = 64
    
    # Initialisation du modèle
    gan = DCGAN(latent_dim=LATENT_DIM, image_size=IMAGE_SIZE)
    gan.generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    
    # Préparation des données
    data_dir = pathlib.Path('./data/images')
    dataset = prepare_dataset(data_dir, BATCH_SIZE, IMAGE_SIZE)
    
    # Entraînement
    train(gan, dataset, EPOCHS, BATCH_SIZE, LATENT_DIM)
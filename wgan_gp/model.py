import tensorflow as tf
from tensorflow.keras import layers

from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS


def build_generator(latent_dim: int = 128) -> tf.keras.Model:
    """
    Generator: latent vector → 128x128 grayscale image in [-1, 1].
    Uses Conv2DTranspose (x5) to go from 4x4 to 128x128.
    """
    model = tf.keras.Sequential([
        layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(latent_dim,)),
        layers.Reshape((4, 4, 512)),
        layers.LayerNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False),
        layers.LayerNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False),
        layers.LayerNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False),
        layers.LayerNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False),
        layers.LayerNormalization(),
        layers.ReLU(),

        # Output layer — tanh keeps values in [-1, 1]
        layers.Conv2DTranspose(
            IMG_CHANNELS, (3, 3), strides=(2, 2),
            padding='same', use_bias=True, activation='tanh',
        ),
    ], name='generator')
    return model


def build_discriminator() -> tf.keras.Model:
    """
    Discriminator (critic): 128x128 image
    No normalization layers (required for WGAN-GP).
    """
    model = tf.keras.Sequential([
        layers.Conv2D(64,  (3, 3), strides=(2, 2), padding='same',
                      input_shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS]),
        layers.LeakyReLU(0.2),

        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),

        layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),

        layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1),
    ], name='discriminator')
    return model
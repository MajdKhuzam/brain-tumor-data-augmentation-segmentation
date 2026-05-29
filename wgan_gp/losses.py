import tensorflow as tf


def gradient_penalty(
    discriminator: tf.keras.Model,
    real_images: tf.Tensor,
    fake_images: tf.Tensor,
) -> tf.Tensor:
    """
    Two-sided gradient penalty (WGAN-GP).
    Interpolates uniformly between real and fake images and penalises the
    squared deviation of the gradient norm from 1.
    """
    batch_size  = tf.shape(real_images)[0]
    real_images = tf.cast(real_images, tf.float32)
    fake_images = tf.cast(fake_images, tf.float32)

    alpha        = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_images + (1.0 - alpha) * fake_images

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm  = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-8) # +eps for stability
    gp    = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


def discriminator_loss(
    real_output: tf.Tensor,
    fake_output: tf.Tensor,
    gp: tf.Tensor,
    lambda_gp: float,
) -> tf.Tensor:
    """Wasserstein discriminator loss + gradient penalty."""
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + lambda_gp * gp


def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """Wasserstein generator loss (maximise critic score on fakes)."""
    return -tf.reduce_mean(fake_output)
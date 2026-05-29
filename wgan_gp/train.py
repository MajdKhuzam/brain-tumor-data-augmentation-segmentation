import os
import numpy as np
import tensorflow as tf

from config import LATENT_DIM, N_CRITIC, LAMBDA_GP, OUTPUT_DIR, SAMPLE_EVERY
from losses import gradient_penalty, discriminator_loss, generator_loss
from visualize import save_sample


# ── Per-step functions ────────────────────────────────────────────────────────

@tf.function
def train_critic(
    images: tf.Tensor,
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    discriminator_optimizer: tf.keras.optimizers.Optimizer,
) -> tf.Tensor:
    noise = tf.random.normal([tf.shape(images)[0], LATENT_DIM])
    with tf.GradientTape() as tape:
        fake_images  = tf.stop_gradient(generator(noise, training=False))
        real_output  = discriminator(images,      training=True)
        fake_output  = discriminator(fake_images, training=True)
        gp           = gradient_penalty(discriminator, images, fake_images)
        disc_loss    = discriminator_loss(real_output, fake_output, gp, LAMBDA_GP)

    grads = tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(
        zip(grads, discriminator.trainable_variables)
    )
    return disc_loss


@tf.function
def train_generator(
    images: tf.Tensor,
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    generator_optimizer: tf.keras.optimizers.Optimizer,
) -> tf.Tensor:
    noise = tf.random.normal([tf.shape(images)[0], LATENT_DIM])
    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        fake_output = discriminator(fake_images, training=True)
        gen_loss    = generator_loss(fake_output)

    grads = tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(
        zip(grads, generator.trainable_variables)
    )
    return gen_loss


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    dataset: tf.data.Dataset,
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    generator_optimizer: tf.keras.optimizers.Optimizer,
    discriminator_optimizer: tf.keras.optimizers.Optimizer,
    epochs: int,
    seed_noise: tf.Tensor,
):
    """
    Train WGAN-GP for *epochs* epochs.

    Returns:
        gen_loss_history  (list[float])
        disc_loss_history (list[float])
    """
    gen_loss_history  = []
    disc_loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_gen_losses  = []
        epoch_disc_losses = []

        for image_batch in dataset:
            # Train critic N_CRITIC times per generator step
            for _ in range(N_CRITIC):
                disc_loss = train_critic(
                    image_batch, generator, discriminator,
                    discriminator_optimizer,
                )

            gen_loss = train_generator(
                image_batch, generator, discriminator,
                generator_optimizer,
            )

            epoch_gen_losses.append(gen_loss.numpy())
            epoch_disc_losses.append(disc_loss.numpy())

        mean_gen  = np.mean(epoch_gen_losses)
        mean_disc = np.mean(epoch_disc_losses)
        gen_loss_history.append(mean_gen)
        disc_loss_history.append(mean_disc)

        print(f'Epoch {epoch:4d} | G loss: {mean_gen:+.4f} | D loss: {mean_disc:+.4f}')

        # Save checkpoint and sample images
        if epoch == 1 or epoch % SAMPLE_EVERY == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, 'generator_model.keras')
            sample_path = os.path.join(OUTPUT_DIR, f'generated_epoch_{epoch:04d}.png')
            generator.save(ckpt_path)
            save_sample(generator, seed_noise, epoch, sample_path)

    # Guaranteed final save
    final_path = os.path.join(OUTPUT_DIR, 'generator_model_final.keras')
    generator.save(final_path)
    print(f'Training complete. Final model saved to {final_path}')

    return gen_loss_history, disc_loss_history
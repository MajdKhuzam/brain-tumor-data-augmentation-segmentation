"""
main.py — WGAN-GP entry point

Run with:
    python main.py
"""
import tensorflow as tf

from config import (
    LATENT_DIM, EPOCHS,
    LR_G, LR_D,
    N_SEED_IMAGES,
)
from data      import load_images, build_dataset
from model     import build_generator, build_discriminator
from train     import train
from visualize import plot_losses, generate_samples


def main():
    # ── Data ─────────────────────────────────────────────────────────────────
    images  = load_images()
    dataset = build_dataset(images)

    # ── Models ───────────────────────────────────────────────────────────────
    generator     = build_generator(LATENT_DIM)
    discriminator = build_discriminator()

    generator.summary()
    discriminator.summary()

    # ── Optimisers ────────────────────────────────────────────────────────────
    generator_optimizer     = tf.keras.optimizers.Adam(LR_G, beta_1=0.0, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(LR_D, beta_1=0.0, beta_2=0.9)

    # ── Fixed noise for progress tracking ────────────────────────────────────
    seed_noise = tf.random.normal([N_SEED_IMAGES, LATENT_DIM])

    # ── Training ──────────────────────────────────────────────────────────────
    g_loss, d_loss = train(
        dataset, generator, discriminator,
        generator_optimizer, discriminator_optimizer,
        epochs=EPOCHS,
        seed_noise=seed_noise,
    )

    # ── Post-training visualisation ───────────────────────────────────────────
    plot_losses(g_loss, d_loss)
    generate_samples(generator, LATENT_DIM, num_images=8)


if __name__ == '__main__':
    main()
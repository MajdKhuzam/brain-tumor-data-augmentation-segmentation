import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def save_sample(
    generator: tf.keras.Model,
    seed_noise: tf.Tensor,
    epoch: int,
    output_path: str | None = None,
) -> None:
    """Generate a 2x8 grid of images from fixed seed noise and save to disk."""
    imgs = generator(seed_noise, training=False)
    imgs = ((imgs + 1.0) / 2.0 * 255.0).numpy().squeeze().astype(np.uint8)

    n = len(imgs)
    cols = min(8, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for ax, img in zip(np.array(axes).flat, imgs):
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.tight_layout()

    save_to = output_path or f'generated_epoch_{epoch:04d}.png'
    plt.savefig(save_to, dpi=100)
    plt.close()


def plot_losses(
    g_losses: list[float],
    d_losses: list[float],
) -> None:
    """Plot generator and discriminator loss curves over training epochs."""
    plt.figure(figsize=(12, 7))
    plt.title('GAN Training Loss', fontsize=18, fontweight='bold', pad=20)

    # Negate so both curves go downward as training improves
    plt.plot([x for x in g_losses], label='Generator Loss',     color='#1f77b4', linewidth=2)
    plt.plot([x for x in d_losses], label='Discriminator Loss', color='#ff7f0e', linewidth=2)

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss',  fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12, loc='upper right')
    plt.show()


def generate_samples(
    generator: tf.keras.Model,
    latent_dim: int,
    num_images: int = 4,
) -> None:
    """Generate and display *num_images* random samples from the generator."""
    noise = tf.random.normal([num_images, latent_dim])
    generated = generator(noise, training=False)
    generated = ((generated + 1) * 127.5).numpy().astype(np.uint8)

    plt.figure(figsize=(num_images * 2.5, 2.5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(np.squeeze(generated[i]), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
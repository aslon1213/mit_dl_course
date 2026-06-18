# deep_learning_mit

My code and experiments following along with MIT's deep learning lectures. Each
folder corresponds to a lecture and contains the notebooks / scripts I wrote
while working through that topic.

## Lectures

| Folder | Topic |
| --- | --- |
| `lecture_9` | Transformers and a ResNet test (`transformer.py`, `transformer.ipynb`, `test_resnet.ipynb`) |
| `lecture_10` | LSTM (`lstm.ipynb`) |
| `lecture_11` | Denoising autoencoder that reconstructs clean images from noisy MNIST digits (`denoising_autoencoder.ipynb`) |
| `lecture_12` | Self-supervised learning: SimCLR contrastive training on CIFAR-10, and a CLIP vs. DINOv2 comparison using embeddings + K-NN for classification (`simCLR_cifar.ipynb`, `compare_clip_dinov2.ipynb`) |
| `lecture_13` | Information bottleneck: reproducing the "information plane" experiment (Shwartz-Ziv & Tishby, 2017) on MNIST to see whether a network compresses during training (`mnist_binning.ipynb`) |
| `lecture_14` | Generative models — see breakdown below |

### Lecture 14 — Generative models

A survey of the main families of generative models, each in its own notebook.

| Notebook | What it does |
| --- | --- |
| `density_estimation.ipynb` | RealNVP normalizing flow on a 2D toy distribution |
| `pixel_cnn.ipynb` | PixelCNN autoregressive image model |
| `gan_generator_mnist.ipynb` | Vanilla GAN generating MNIST digits |
| `dcgan_cifar10.ipynb` | DCGAN on CIFAR-10 |
| `wgan-gp_cifar10.ipynb` | Wasserstein GAN with gradient penalty on CIFAR-10 |
| `ddpm_mnist.ipynb` | Denoising diffusion probabilistic model (DDPM) on MNIST |
| `classifier_free_guidance.ipynb` | Classifier-free guidance for conditional diffusion |
| `latent_diffusion_model_vqvae.ipynb` | VQ-VAE that encodes images into the latent space for latent diffusion |
| `latent_diffusion_model_ddpm_class.ipynb` | Class-conditioned latent DDPM (UNet + linear noise scheduler) |
| `latent_diffusion_model_ddpm_image_cond.ipynb` | Image-conditioned latent DDPM |

Shared building blocks live in `lecture_14/blocks/` (`ddpm_blocks.py`,
`vqvae_blocks.py`) and `lecture_14/models/` (`vqvae.py`).

## Setup

This project uses [uv](https://github.com/astral-sh/uv) and Python 3.13.

```bash
uv sync
```

Datasets (MNIST, CIFAR-10, CelebA-HQ) are downloaded into each lecture folder on
first run and are git-ignored.

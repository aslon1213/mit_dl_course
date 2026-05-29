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

## Setup

This project uses [uv](https://github.com/astral-sh/uv) and Python 3.13.

```bash
uv sync
```

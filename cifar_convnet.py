# Testing out Jax by developing a super simple convnet to classify the cifar10 data. Hoping to learn
# jax, haiku, and optax, but also refresh on python classes and learn some new software engineering
# in python in general.

from absl import app

from typing import Iterator, Mapping, Tuple

import optax as opt
import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Following the Deepmind examples
Batch = Mapping[str, np.ndarray]

def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,) -> Iterator[Batch]:
  """Loads the dataset as a generator of batches."""
  ds = tfds.load("cifar10:3.*.*", split=split).cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds))

def main(_):
  train = load_dataset("train", is_training=True, batch_size=32) 
  print(next(train)['image'].shape)

  figure = plt.Figure()

  plt.imshow(np.squeeze(next(train)['image'][0, :, :, :]))

  plt.show()

if __name__ == "__main__":
  app.run(main)
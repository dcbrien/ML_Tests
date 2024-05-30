# Testing out Jax by developing a super simple convnet to classify the cifar10 data. Hoping to learn
# jax, haiku, and optax, but also refresh on python classes and learn some new software engineering
# in python in general.

from absl import app

from typing import Iterator, Mapping, Tuple

import optax 
import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow_datasets as tfds

import numpy as np

#matplotlib.use('TkAgg')

# Following the Deepmind examples - Batch is just a dictionary, so define it 
# as such a type
Batch = Mapping[str, np.ndarray]

def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,) -> (Iterator[Batch], Batch):
  """Loads the dataset as a generator of batches."""
  ds, ds_info = tfds.load("cifar10:3.*.*", split=split, with_info=True)
  ds = ds.cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds)), ds_info

def net_fn(batch: Batch) -> jnp.ndarray:
  # normalize the images
  x = batch["image"].astype(jnp.float32) / 255.0 - 128.0

  # The most simple network - Flatten the image and pass through some
  # convolutions, a mlp, and output the 10 categories as logits
  cnet = hk.Sequential([
    hk.Conv2D(64, (3, 3), 1, 1, "SAME"), jax.nn.relu,
    hk.MaxPool((2,2), 2, "SAME"),
    hk.Conv2D(64, (3, 3), 1, 1, "SAME"), jax.nn.relu,
    hk.MaxPool((2,2), 2, "SAME"),
    hk.Conv2D(64, (3, 3), 1, 1, "SAME"), jax.nn.relu,
    hk.Flatten(),
    hk.Linear(64), jax.nn.relu,
    hk.Linear(64), jax.nn.relu,
    hk.Linear(10),
  ])

  return cnet(x)


# Evaluation metric (classification accuracy).
@jax.jit
def accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
  predictions = net.apply(params, batch)
  return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])



# def main(_):

# if __name__ == "__main__":
#   app.run(main)
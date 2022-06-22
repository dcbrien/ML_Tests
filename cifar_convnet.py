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
    batch_size: int,) -> Iterator[Batch]:
  """Loads the dataset as a generator of batches."""
  ds = tfds.load("cifar10:3.*.*", split=split).cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds))

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

# Training loss (cross-entropy). - right from the examples. pretty simple function
def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
  """Compute the loss of the network, including L2."""
  logits = net.apply(params, batch)
  labels = jax.nn.one_hot(batch["label"], 10)

  l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
  softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
  softmax_xent /= labels.shape[0]

  return softmax_xent + 1e-4 * l2_loss

# Evaluation metric (classification accuracy).
@jax.jit
def accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
  predictions = net.apply(params, batch)
  return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

@jax.jit
def update(
    params: hk.Params,
    opt_state: optax.OptState,
    batch: Batch,
) -> Tuple[hk.Params, optax.OptState]:
  """Learning rule (stochastic gradient descent)."""
  grads = jax.grad(loss)(params, batch)
  updates, opt_state = opt.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  return new_params, opt_state



# def main(_):

# if __name__ == "__main__":
#   app.run(main)
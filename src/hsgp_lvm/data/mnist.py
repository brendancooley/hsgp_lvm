import jax
import jax.numpy as jnp
from numpyro.examples.datasets import MNIST, load_dataset


def load_data(
    tr_batch_size: int, test_batch_size: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    train_init, train_fetch = load_dataset(
        MNIST, shuffle=False, batch_size=tr_batch_size
    )
    test_init, test_fetch = load_dataset(
        MNIST, split="test", shuffle=False, batch_size=test_batch_size
    )

    _, train_idx = train_init()
    _, test_idx = test_init()

    img_tr, y_tr = train_fetch(0, train_idx)
    img_test, y_test = test_fetch(0, test_idx)

    img = jnp.concatenate([img_tr, img_test], axis=0)
    X = img.reshape((img.shape[0], -1))  # [0, 1]-valued observations
    y = jnp.concatenate([y_tr, y_test], axis=0).astype(
        jnp.int8
    )  # {0, ..., 9}-valued labels
    mask = jnp.concatenate(
        [jnp.ones_like(y_tr), jnp.zeros_like(y_test)], axis=0
    ).astype(jnp.bool)  # boolean mask for labels in the training set
    return X, y, mask

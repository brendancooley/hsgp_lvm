import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers


@jax.tree_util.register_pytree_node_class
class PPCA:
    def __init__(
        self, latent_dim: int, num_class: int, reconstruction_w: float = 0.25, **kwargs
    ):
        self.latent_dim = latent_dim
        self.num_class = num_class
        self.reconstruction_w = reconstruction_w

    def model(
        self,
        X: jnp.Array,
        y: jax.Array,
        mask: jax.Array,
        subsample_size: int | None = None,
    ):
        out_dim = X.shape[1]

        beta_img = numpyro.sample(
            "beta_img", dist.Normal(0, 1), sample_shape=(self.latent_dim, out_dim)
        )
        beta_cls = numpyro.sample(
            "beta_cls",
            dist.Normal(0, 1),
            sample_shape=(self.latent_dim, self.num_class),
        )
        with numpyro.plate("data", X.shape[0], subsample_size=subsample_size) as ind:
            X_batch = X[ind]
            y_batch = y[ind]
            mask_batch = mask[ind]

            Z = numpyro.sample(
                "Z", dist.Normal(0, 1).expand([subsample_size, self.latent_dim])
            )

            f_img = Z @ beta_img
            p_img = numpyro.deterministic("p_img", jax.scipy.special.expit(f_img))
            with handlers.scale(
                scale=self.reconstruction_w  # TODO how to set this in a principled way?
            ):
                numpyro.sample(
                    "obs", dist.BernoulliProbs(p_img).to_event(1), obs=X_batch
                )

            f_cls = Z @ beta_cls
            p_cls = numpyro.deterministic("p_cls", jax.nn.softmax(f_cls))
            with handlers.mask(mask=mask_batch):
                numpyro.sample("y", dist.CategoricalProbs(p_cls), obs=y_batch)

    def tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = self.latent_dim  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

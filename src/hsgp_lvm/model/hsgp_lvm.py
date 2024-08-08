import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.contrib.hsgp.approximation import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_squared_exponential,
)

from hsgp_lvm.model.config import ModelConfig


class HSGPLVMConfig(ModelConfig):
    ell: int
    m: int


def hsgp(config: HSGPLVMConfig, out_dim: int, prefix: str):
    beta_shape = (config.m**config.latent_dim, out_dim)
    with handlers.scope(prefix=prefix, divider="_"):
        alpha_mean = numpyro.sample("alpha_mean", dist.Normal(0, 1))
        alpha_std = numpyro.sample("alpha_std", dist.LogNormal(0, 1))
        alpha = numpyro.sample(
            "alpha", dist.LogNormal(alpha_mean, alpha_std), sample_shape=(out_dim,)
        )

        length_mean = numpyro.sample("length_mean", dist.Normal(0, 1))
        length_std = numpyro.sample("length_std", dist.LogNormal(0, 1))
        length = numpyro.sample(
            "length",
            dist.LogNormal(length_mean, length_std),
            sample_shape=(out_dim, config.latent_dim),
        )

        spd = jnp.sqrt(
            diag_spectral_density_squared_exponential(
                alpha=alpha,
                length=length,
                ell=config.ell,
                m=config.m,
                dim=config.latent_dim,
            )
        )
        beta = numpyro.sample("beta", dist.Normal(0, 1), sample_shape=beta_shape)
    return spd * beta


@jax.tree_util.register_pytree_node_class
class HSGPLVM:
    def __init__(
        self,
        config: HSGPLVMConfig,
        **kwargs,
    ):
        self.config = config

    def model(
        self,
        X: jax.Array,
        y: jax.Array,
        mask: jax.Array,
        subsample_size: int | None = None,
    ):
        img_out_dim = X.shape[1]

        hsgp_img = hsgp(self.config, img_out_dim, prefix="img")
        hsgp_cls = hsgp(self.config, self.config.num_class, prefix="cls")
        with numpyro.plate("data", X.shape[0], subsample_size=subsample_size) as ind:
            X_batch = X[ind]
            y_batch = y[ind]
            mask_batch = mask[ind]
            Z = numpyro.sample(
                "Z", dist.Normal(0, 1).expand([self.config.latent_dim]).to_event(1)
            )
            eig_f = eigenfunctions(x=Z, ell=self.config.ell, m=self.config.m)

            # reconstruct images
            f_img = eig_f @ hsgp_img
            p_img = numpyro.deterministic("p_img", jax.scipy.special.expit(f_img))
            with handlers.scale(
                scale=self.config.reconstruction_w  # TODO how to set this in a principled way?
            ):
                numpyro.sample(
                    "obs", dist.BernoulliProbs(p_img).to_event(1), obs=X_batch
                )

            # classify images
            f_cls = eig_f @ hsgp_cls
            p_cls = numpyro.deterministic("p_cls", jax.nn.softmax(f_cls, axis=-1))
            with handlers.mask(mask=mask_batch):
                numpyro.sample("y", dist.CategoricalProbs(p_cls), obs=y_batch)

    def tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = (self.config,)  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

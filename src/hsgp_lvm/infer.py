from jax import random
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.elbo import Trace_ELBO
from numpyro.infer.initialization import init_to_median
from numpyro.infer.svi import SVI
from numpyro.optim import Adam
from optax import linear_onecycle_schedule


def fit_svi(
    seed: int,
    model: callable,
    guide: callable,
    num_steps: int = 1000,
    peak_lr: float = 0.01,
    progress_bar: bool = True,
    use_scheduler: bool = False,
    **model_kwargs,
):
    if use_scheduler:
        lr = linear_onecycle_schedule(num_steps, peak_lr)
    else:
        lr = peak_lr
    svi = SVI(model, guide, Adam(lr), Trace_ELBO())
    return svi.run(
        random.PRNGKey(seed), num_steps, progress_bar=progress_bar, **model_kwargs
    )

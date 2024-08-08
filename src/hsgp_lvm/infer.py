from jax import random
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.elbo import Trace_ELBO
from numpyro.infer.initialization import init_to_median
from numpyro.infer.svi import SVI
from numpyro.optim import Adam
from optax import linear_onecycle_schedule
from pydantic import BaseModel, model_validator


class SVIConfig(BaseModel):
    num_steps: int = 50_000
    use_scheduler: bool = True
    peak_lr: float | None = 0.01
    lr: float | None = 0.01
    progress_bar: bool = True

    @model_validator(mode="after")
    def validate_lr(self):
        if self.use_scheduler and self.peak_lr is None:
            raise ValueError("peak_lr must be provided if use_scheduler is True")
        if not self.use_scheduler and self.lr is None:
            raise ValueError("lr must be provided if use_scheduler is False")


def fit_svi(
    seed: int,
    model: callable,
    guide: callable,
    svi_config: SVIConfig,
    **model_kwargs,
):
    if svi_config.use_scheduler:
        lr = linear_onecycle_schedule(svi_config.num_steps, svi_config.peak_lr)
    else:
        lr = svi_config.lr
    svi = SVI(model, guide, Adam(lr), Trace_ELBO())
    return svi.run(
        random.PRNGKey(seed),
        svi_config.num_steps,
        progress_bar=svi_config.progress_bar,
        **model_kwargs,
    )

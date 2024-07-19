import logging
from typing import Literal

import modal
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_median

from hsgp_lvm.data.mnist import load_data
from hsgp_lvm.infer import fit_svi
from hsgp_lvm.model.hsgp_lvm import HSGPLVM
from hsgp_lvm.model.ppca import PPCA

app = modal.App()
image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(
    "requirements.txt"
)


@app.function(image=image)
def fit(seed, model, *args, **kwargs):
    guide = AutoNormal(model, init_loc_fn=init_to_median(num_samples=50))
    return fit_svi(seed=seed, model=model, guide=guide, *args, **kwargs)


@app.local_entrypoint()
def main(
    model_name: str = "hsgp_lvm",
    latent_dim: int = 2,
    tr_batch_size: int = 60_000,
    test_batch_size: int = 10_000,
    ell: int = 5,
    m: int = 10,
    num_steps: int = 10,
):
    logging.basicConfig(level=logging.INFO)
    model_cls = HSGPLVM if model_name == "hsgp_lvm" else PPCA
    logging.info(f"Insantiating {model_cls.__name__} model.")
    model = model_cls(latent_dim=latent_dim, num_class=10, ell=ell, m=m)
    logging.info("Loading data.")
    X, y, mask = load_data(tr_batch_size, test_batch_size)
    logging.info("Fitting model.")
    svi_res = fit.remote(
        seed=0,
        model=model.model,
        X=X,
        y=y,
        mask=mask,
        use_scheduler=True,
        num_steps=num_steps,
        peak_lr=0.01,
        # init_params=None if init_params is None else init_params.params,
        subsample_size=1_000,
    )
    print("done")
    # TODO handle save/load

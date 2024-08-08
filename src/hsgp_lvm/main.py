import json
import logging
from typing import Callable, Literal

import modal
from dotenv import load_dotenv
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_median
from pydantic import BaseModel

from hsgp_lvm.data.mnist import load_data
from hsgp_lvm.infer import SVIConfig, fit_svi
from hsgp_lvm.io import ModelSaver
from hsgp_lvm.model.config import ModelConfig
from hsgp_lvm.model.hsgp_lvm import HSGPLVM, HSGPLVMConfig
from hsgp_lvm.model.ppca import PPCA

app = modal.App()
image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(
    "requirements.txt"
)


@app.function(image=image)
def fit(seed: int, model: Callable[..., None], svi_config: SVIConfig, **model_kwargs):
    guide = AutoNormal(model, init_loc_fn=init_to_median(num_samples=50))
    return fit_svi(
        seed=seed, model=model, guide=guide, svi_config=svi_config, **model_kwargs
    )


@app.local_entrypoint()
def main(
    model_name: str = "hsgp_lvm",
    tr_batch_size: int = 60_000,
    test_batch_size: int = 10_000,
    model_config_path: str | None = None,
    svi_config_path: str | None = None,
):
    # TODO training configurables
    if model_config_path is None:
        model_config_path = f"config/{model_name}.json"
    if svi_config_path is None:
        svi_config_path = f"config/svi.json"
    model_config_cls = ModelConfig if model_name == "ppca" else HSGPLVMConfig
    with open(model_config_path) as f:
        model_config = model_config_cls(**json.load(f))
    with open(svi_config_path) as f:
        svi_config = SVIConfig(**json.load(f))
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    model_cls = HSGPLVM if model_name == "hsgp_lvm" else PPCA
    logging.info(f"Insantiating {model_cls.__name__} model.")
    model = model_cls(config=model_config)
    logging.info("Loading data.")
    X, y, mask = load_data(tr_batch_size, test_batch_size)
    logging.info("Fitting model.")
    svi_res = fit.remote(
        seed=0,
        model=model.model,
        svi_config=svi_config,
        X=X,
        y=y,
        mask=mask,
    )
    logging.info("Saving SVI result.")
    saver = ModelSaver(model_name)
    saver.save_svi_result(svi_res)
    saver.save_model_config(model_config)
    saver.save_svi_config(svi_config)
    print("done")

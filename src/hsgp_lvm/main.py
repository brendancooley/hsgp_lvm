import json
import logging
from typing import Callable, Literal

import jax
import modal
from dotenv import load_dotenv
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_median
from pydantic import BaseModel

from hsgp_lvm.data.mnist import load_data
from hsgp_lvm.enum import MODEL_NAME
from hsgp_lvm.infer import InferConfig, fit_svi
from hsgp_lvm.model.config import ModelConfig
from hsgp_lvm.model.hsgp_lvm import HSGPLVM, HSGPLVMConfig
from hsgp_lvm.model.ppca import PPCA
from hsgp_lvm.results import ModelResult

app = modal.App()
image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(
    "requirements.txt"
)


@app.function(image=image)
def fit(
    seed: int, model: Callable[..., None], infer_config: InferConfig, **model_kwargs
):
    guide = AutoNormal(model, init_loc_fn=init_to_median(num_samples=50))
    return fit_svi(
        seed=seed, model=model, guide=guide, infer_config=infer_config, **model_kwargs
    )


@app.local_entrypoint()
def main(
    model_name: MODEL_NAME = "hsgp_lvm",
    tr_batch_size: int = 60_000,
    test_batch_size: int = 10_000,
    model_config_path: str | None = None,
    infer_config_path: str | None = None,
    local: bool = False,
):
    load_dotenv(override=True)

    if model_config_path is None:
        model_config_path = f"config/{model_name}.json"
    if infer_config_path is None:
        infer_config_path = f"config/infer.json"
    model_config_cls = ModelConfig if model_name == "ppca" else HSGPLVMConfig
    with open(model_config_path) as f:
        model_config = model_config_cls(**json.load(f))
    with open(infer_config_path) as f:
        infer_config = InferConfig(**json.load(f))

    logging.basicConfig(level=logging.INFO)
    model_cls = HSGPLVM if model_name == "hsgp_lvm" else PPCA
    logging.info(f"Insantiating {model_cls.__name__} model.")
    model = model_cls(config=model_config)
    logging.info("Loading data.")
    X, y, mask = load_data(tr_batch_size, test_batch_size)
    logging.info("Fitting model.")
    fit_kwargs = {
        "seed": 0,
        "model": model.model,
        "infer_config": infer_config,
        "X": X,
        "y": y,
        "mask": mask,
    }
    svi_res = fit.local(**fit_kwargs) if local else fit.remote(**fit_kwargs)
    logging.info("Saving SVI result.")
    result = ModelResult(model_name, svi_res, model_config, infer_config)
    result.save()
    print("done")
    # TODO metrics


if __name__ == "__main__":
    jax.config.update("jax_platforms", "cpu")
    main(local=True)

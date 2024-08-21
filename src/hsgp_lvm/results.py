import datetime
import json
import logging
import os
import pickle
from pathlib import Path
from jax import random

import s3fs
from numpyro.infer.svi import SVIRunResult
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoGuide

from hsgp_lvm.infer import InferConfig
from hsgp_lvm.model.config import ModelConfig


class ModelResult:
    def __init__(
        self,
        model_name: str,
        guide_cls: type[AutoGuide],
        svi_res: SVIRunResult,
        model_config: ModelConfig,
        infer_config: InferConfig,
        timestamp: str | None = None,
    ):
        self.model_name = model_name
        self.guide_cls = guide_cls
        self.svi_res = svi_res
        self.model_config = model_config
        self.infer_config = infer_config

        if timestamp is None:
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        else:
            self.timestamp = timestamp

        self.s3 = s3fs.S3FileSystem()

    @classmethod
    def _prefix(cls, model_name: str, timestamp: str):
        return f"{os.environ.get('S3_BUCKET_NAME')}/{model_name}/{timestamp}"

    def _generate_key(self, fname: str) -> str:
        return f"{self._prefix(self.model_name, self.timestamp)}/{fname}"

    @classmethod
    def load(cls, model_name: str, timestamp: str | None = None):
        s3 = s3fs.S3FileSystem()
        if timestamp is None:
            timestamp = cls.available_models(model_name)[-1].split("/")[-1]
        prefix = cls._prefix(model_name, timestamp)
        svi_res = pickle.load(s3.open(f"{prefix}/svi_result.pkl", "rb"))
        guide_cls = pickle.load(s3.open(f"{prefix}/guide_cls.pkl", "rb"))  # TODO try to reconstitute so we can avoid heavy save, or look into why the file is so large
        with s3.open(f"{prefix}/model_config.json") as f:
            model_config = ModelConfig(**json.load(f))
        with s3.open(f"{prefix}/infer_config.json") as f:
            infer_config = InferConfig(**json.load(f))
        return cls(model_name, guide_cls, svi_res, model_config, infer_config, timestamp)

    def save(self):
        for obj, fname in zip(
            [self.guide_cls, self.svi_res, self.model_config, self.infer_config],
            ["guide_cls.pkl", "svi_result.pkl", "model_config.json", "infer_config.json"],
        ):
            key = self._generate_key(fname)
            with self.s3.open(key, "wb" if key.endswith(".pkl") else "w") as f:
                logging.info(f"Saving {fname} to s3://{key}")
                if key.endswith(".json"):
                    json.dump(obj.model_dump(), f)
                else:
                    pickle.dump(obj, f)

    @classmethod
    def available_models(cls, model_name: str):
        s3 = s3fs.S3FileSystem()
        return sorted(s3.ls(path=Path(os.environ.get("S3_BUCKET_NAME")) / model_name))

    # TODO add purge method

    def posterior(self, model: callable, seed: int = 0, num_samples: int = 500):
        guide = self.guide_cls(model)
        return Predictive(model=self.guide_cls, params=self.svi_res.params, num_samples=num_samples)(random.PRNGKey(seed))
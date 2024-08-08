import datetime
import json
import logging
import os
import pickle

import s3fs
from numpyro.infer.svi import SVIRunResult

from hsgp_lvm.infer import SVIConfig
from hsgp_lvm.model.config import ModelConfig


class ModelSaver:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.s3 = s3fs.S3FileSystem()

    def _generate_key(self, fname: str) -> str:
        return f"{os.environ.get('S3_BUCKET_NAME')}/{self.model_name}/{self.timestamp}/{fname}"

    def save_svi_result(self, svi_res: SVIRunResult):
        key = self._generate_key("svi_result.pkl")
        with self.s3.open(key, "wb") as f:
            logging.info(f"Saving SVI result to s3://{key}")
            pickle.dump(svi_res, f)

    def save_model_config(self, model_config: ModelConfig):
        key = self._generate_key("model_config.json")
        with self.s3.open(key, "w") as f:
            logging.info(f"Saving model config to s3://{key}")
            json.dump(model_config.model_dump(), f)

    def save_svi_config(self, svi_config: SVIConfig):
        key = self._generate_key("svi_config.json")
        with self.s3.open(key, "w") as f:
            logging.info(f"Saving svi config to s3://{key}")
            json.dump(svi_config.model_dump(), f)

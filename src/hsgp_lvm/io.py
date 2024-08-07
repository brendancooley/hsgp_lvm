import datetime
import logging
import os
import pickle

import s3fs
from numpyro.infer.svi import SVIRunResult


def save_svi_result(svi_res: SVIRunResult, dir_prefix: str):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    s3 = s3fs.S3FileSystem()
    key = f"{os.environ.get('S3_BUCKET_NAME')}/{dir_prefix}/{timestamp}/model.pkl"
    with s3.open(key, "wb") as f:
        logging.info(f"Saving SVI result to s3://{key}")
        pickle.dump(svi_res, f)

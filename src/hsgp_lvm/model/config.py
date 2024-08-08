from pydantic import BaseModel


class ModelConfig(BaseModel):
    latent_dim: int
    num_class: int
    reconstruction_w: float

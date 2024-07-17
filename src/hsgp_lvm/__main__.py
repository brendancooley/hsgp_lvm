from typing import Literal

from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_median
from tap import Tap

from hsgp_lvm.data.mnist import load_data
from hsgp_lvm.infer import fit_svi
from hsgp_lvm.model.hsgp_lvm import HSGPLVM
from hsgp_lvm.model.ppca import PPCA


class Args(Tap):
    model: Literal["ppca", "hsgp_lvm"] = "hsgp_lvm"
    latent_dim: int = 2
    tr_batch_size: int = 60_000
    test_batch_size: int = 10_000
    ell: int = 5
    m: int = 10
    num_steps: int = 10_000


def main(args: Args):
    model_cls = HSGPLVM if args.model == "hsgp_lvm" else PPCA
    model = model_cls(latent_dim=args.latent_dim, num_class=10, ell=args.ell, m=args.m)
    X, y, mask = load_data(args.tr_batch_size, args.test_batch_size)
    guide = AutoNormal(model.model, init_loc_fn=init_to_median(num_samples=50))
    svi_res = fit_svi(
        seed=0,
        model=model.model,
        guide=guide,
        X=X,
        y=y,
        mask=mask,
        use_scheduler=True,
        num_steps=args.num_steps,
        peak_lr=0.01,
        # init_params=None if init_params is None else init_params.params,
        subsample_size=1_000,
    )
    # TODO onto modal, handle save/load

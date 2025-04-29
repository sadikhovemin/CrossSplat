from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossDepthMaeCfg:
    weight: float


@dataclass
class LossDepthMaeCfgWrapper:
    depth_mae: LossDepthMaeCfg


class LossDepthMae(Loss[LossDepthMaeCfg, LossDepthMaeCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        delta = prediction.depth - batch["target"]["depth"]
        return self.cfg.weight * delta.abs().mean()
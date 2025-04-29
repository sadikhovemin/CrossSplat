from .loss import Loss
from .loss_depth_mse import LossDepthMse, LossDepthMseCfgWrapper
from .loss_depth_mae import LossDepthMae, LossDepthMaeCfgWrapper
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossDepthMseCfgWrapper: LossDepthMse,
    LossDepthMaeCfgWrapper: LossDepthMae,
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossDepthMseCfgWrapper | LossDepthMaeCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
